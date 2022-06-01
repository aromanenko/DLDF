from architectures import *
from functions import *
from generate_features import read_json
from ipywidgets import interact, interactive 
from IPython.display import display
import json
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from pytorch_forecasting.data import (
    TimeSeriesDataSet,
    GroupNormalizer
)

import pytorch_lightning as pl
from pytorch_forecasting.metrics import QuantileLoss, SMAPE
from pytorch_forecasting.models import TemporalFusionTransformer, NBeats, DeepAR


def choose_model_(model, model_config_filename):
    config = read_json(model_config_filename)
    config['model'] = model
    with open(model_config_filename, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Model: {model}")
    

def choose_model(model_config_filename):
    print('Choose model:')
    display(interactive(
        lambda model : choose_model_(model, model_config_filename),
        model=['LSTM', 'GRU', 'TFT', 'DeepAR', 'NBeats']
    ))
    
    
def pf_preprocess(df_train, df_test, config):
    df_train = df_train.reset_index()
    df_test = df_test.reset_index()
    df_train["time_idx"] = df_train[config['date_col']].dt.dayofyear + (df_train[config['date_col']].dt.year - 2015) * 365
    df_test["time_idx"] = df_test[config['date_col']].dt.dayofyear + (df_test[config['date_col']].dt.year - 2015) * 365
    min_time_idx = df_train["time_idx"].min()
    df_train["time_idx"] -= min_time_idx
    df_test["time_idx"] -= min_time_idx
    for feature in config['categorical'] + config['id_cols']:
        df_train[feature] = df_train[feature].astype(str)
        df_test[feature] = df_test[feature].astype(str)
    df_test[config['target']] = 0
    return df_train, df_test


def pf_get_dataloader(df_train, df_test, config):
    dates = df_test.reset_index()[config['date_col']]
    max_prediction_length = (dates.max() - dates.min()).days + 1
    max_encoder_length = 60
    params = {}
    new_groups = []
    for col in config['id_cols']:
        if len(set(df_test[col]) - set(df_train[col])) != 0:
            new_groups.append(col)
    if config['model'] == 'NBeats':
        params = {
            "categorical_encoders": {x: NaNLabelEncoder().fit(df_train[x]) for x in new_groups},
            "time_varying_unknown_reals": [config['target']]
        }
    if config['model'] == 'DeepAR':
        params = {
            "time_varying_known_categoricals": config['categorical'],
            "time_varying_known_reals": ["time_idx"] + config['real'] + \
            df_train.columns[df_train.columns.str.contains('lag')].values.tolist(),
            "categorical_encoders": {x: NaNLabelEncoder().fit(df_train[x]) for x in new_groups},
            "time_varying_unknown_reals": [config['target']]
        }
    if config['model'] == 'TFT':
        params = {
            "min_encoder_length": 0,
            "min_prediction_length": 1,
            "static_categoricals": config['id_cols'],
            "static_reals": [],
            "categorical_encoders": {x: NaNLabelEncoder(add_nan=True) for x in new_groups},
            "time_varying_known_categoricals": config['categorical'],
            "variable_groups": {},
            "time_varying_known_reals": ["time_idx"] + config['real'] + \
            df_train.columns[df_train.columns.str.contains('lag')].values.tolist(),
            "time_varying_unknown_categoricals": [],
            "target_normalizer": GroupNormalizer(
                groups= config['id_cols']
            ),
            "add_relative_time_idx": True,
            "add_target_scales": True,
            "add_encoder_length": True,
            "allow_missing_timesteps": True
        }
    training = TimeSeriesDataSet(        
        df_train,
        time_idx="time_idx",
        target=config['target'],
        group_ids=config['id_cols'],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        **params
    )

    batch_size = config['batch_size']
    train_dataloader = training.to_dataloader(
        train=True, batch_size=batch_size, num_workers=0
    )
    return training, train_dataloader
    
    
def pf_get_model(training, config):
    models_dict = {'TFT': TemporalFusionTransformer,
                   'DeepAR' : DeepAR,
                   'NBeats' : NBeats}
    params = {
        "dataset": training,
        "learning_rate": config['learning_rate'],
        "dropout": config['dropout'],
        "weight_decay": config['weight_decay']
    }
    if config['model'] == 'DeepAR':
        params["hidden_size"] = config['num_hidden_units']
    if config['model'] == 'TFT':
        params["hidden_size"] = config['num_hidden_units']
        params["attention_head_size"] = config['attention']
        params["hidden_continuous_size"] = config['hidden_continuous_size']
        params["loss"] = SMAPE()
    model = models_dict[config['model']].from_dataset(**params)
    return model


def fit_predict(model, training, train_dataloader, df_train, df_test, config):
    trainer = pl.Trainer(max_epochs=config['epochs'],
                         gpus=int(torch.cuda.is_available()),
                         gradient_clip_val=config['gradient_clip_val'])
    trainer.fit(model, train_dataloaders=train_dataloader)
    if config['model'] == 'TFT':
        val_dataloader = df_test
    else:
        training_cutoff = df_train["time_idx"].max()
        validation = TimeSeriesDataSet.from_dataset(training, pd.concat([df_train, df_test]).reset_index(), min_prediction_idx=training_cutoff + 1)
        val_dataloader = validation.to_dataloader(train=False, batch_size=config['batch_size'])
    pred, ind = model.predict(val_dataloader, return_index=True)
    res_test = df_test[config['id_cols'] + [config['date_col']]].astype(str)
    res_test['pred'] = None
    for i in range(pred.shape[0]):
        store, sku = ind.iloc[i][config['id_cols'][0]], ind.iloc[i][config['id_cols'][1]]
        res_test.loc[(res_test[config['id_cols'][0]] == store) & (res_test[config['id_cols'][1]] == sku), 'pred'] = pred[i].numpy()
    return res_test


def fix_pred_format(res_test, config):
    res_test[config['date_col']] = pd.to_datetime(res_test[config['date_col']])
    res_test[config['id_cols'][0]] = res_test[config['id_cols'][0]].astype(int)
    res_test[config['id_cols'][1]] = res_test[config['id_cols'][1]].astype(int)
    res_test = res_test.sort_values(config['id_cols'] + [config['date_col']]).reset_index()
    res_test = res_test.rename(columns={'pred': config['target']})
    return res_test


def fit_secondary_model(data_lagged_features, config):
    print('Biulding secondary model:')
    df_train, df_test, target_mean, target_stdev, features = preprocess(data_lagged_features,
                                                                        config['target'],
                                                                        config['id_cols'],
                                                                        config['date_col'])
    train_loader, test_loader, train_eval_loader = get_dataloaders(df_train, df_test,
                                                                   config['target'], features, config["batch_size"])
    model = ShallowRegressionLSTM(len(features), config['num_hidden_units'], config['dropout'])
    loss_function = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['gamma'])
    epochs = config['epochs']
    train(df_train, epochs, train_loader, train_eval_loader, model,
          loss_function, optimizer, scheduler, target_mean, target_stdev, eval_=False)
    df_out = get_df_out(model, df_train, df_test, train_eval_loader, test_loader, config['target'], target_mean, target_stdev)
    ans = df_out[df_out[config['target']].isna()].drop(config['target'], axis=1).rename(
        columns={'Model forecast' : config['target']}).sort_values(
        config['id_cols'] + [config['date_col']]).reset_index()[[config['target']]]
    return ans
    
    
def pipeline(data_lagged_features, config):
    model_name = config['model']
    target = config['target']
    id_cols = config['id_cols']
    date_col = config['date_col']
    scl = (model_name in ['LSTM', 'GRU']) * 2
    df_train, df_test, target_mean, target_stdev, features = preprocess(data_lagged_features, target, id_cols, date_col, scl)
    if model_name in ['LSTM', 'GRU']:
        train_loader, test_loader, train_eval_loader = get_dataloaders(df_train, df_test, target, features, config["batch_size"])
        models_dict = {'LSTM': ShallowRegressionLSTM, 'GRU': GRU}
        model = models_dict[config['model']](len(features), config['num_hidden_units'], config['dropout'])
        loss_function = nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config['gamma'])
        epochs = config['epochs']
        print('Training:')
        train(df_train, epochs, train_loader, train_eval_loader, model,
              loss_function, optimizer, scheduler, target_mean, target_stdev)
        df_out = get_df_out(model, df_train, df_test, train_eval_loader, test_loader, target, target_mean, target_stdev)
        ans = df_out[df_out[config['target']].isna()].drop(target, axis=1).rename(
            columns={'Model forecast' : target}).sort_values(
            config['id_cols'] + [config['date_col']]).reset_index()[config['id_cols'] + [config['date_col']] + [config['target']]]
        return ans
    if model_name in ['TFT', 'DeepAR', 'NBeats']:
        df_train, df_test = pf_preprocess(df_train, df_test, config)
        training, train_dataloader = pf_get_dataloader(df_train, df_test, config)
        model = pf_get_model(training, config)
        res_test = fit_predict(model, training, train_dataloader, df_train, df_test, config)
        res_test = fix_pred_format(res_test, config)
        ans2 = fit_secondary_model(data_lagged_features, config)
        res_test.loc[res_test[target].isna(), target] = ans2[res_test[target].isna()][target]
        return res_test[config['id_cols'] + [config['date_col']] + [config['target']]]