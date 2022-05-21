import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import timedelta
from copy import deepcopy
import warnings, pylab
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
import random
import torch
import os
import seaborn as sns
from tqdm.contrib.itertools import product
from architectures import *

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

warnings.filterwarnings('ignore')
sns.set_style("darkgrid")
pd.plotting.register_matplotlib_converters()
sns.mpl.rc("figure", figsize=(25, 5))
sns.mpl.rc("font", size=14)


def scale(df_train, df_test, target, scl):
    if scl == 0:
        return 0, 1
    elif scl == 1:
        cols = df_train.drop(target, axis=1).columns
        target_mean = 0
        target_stdev = 1
    else:
        cols = df_train.columns
        target_mean = df_train[target].mean()
        target_stdev = df_train[target].std()
    print('Scaling:')
    for c in tqdm_notebook(cols):
        mean = df_train[c].mean()
        stdev = df_train[c].std()
        df_train[c] = (df_train[c] - mean) / stdev
        df_test[c] = (df_test[c] - mean) / stdev
    return target_mean, target_stdev

def preprocess(data_lagged_features, target, id_cols, date_col, scl=2):
    print('Preprocessing:')
    data_lagged_features[date_col] = pd.to_datetime(data_lagged_features[date_col])
    for c, sku, store in product(data_lagged_features.drop([date_col, target], axis=1).columns, 
                                 data_lagged_features[id_cols[0]].unique(), data_lagged_features[id_cols[1]].unique()):
        cond = (data_lagged_features[id_cols[0]] == sku) & (data_lagged_features[id_cols[1]] == store)
        data_lagged_features.loc[cond, c] = data_lagged_features[cond][c].fillna(data_lagged_features[cond][c].mean())
    for id_ in id_cols:
        for c in data_lagged_features.drop([date_col, target], axis=1).columns:
            for sku in data_lagged_features[id_].unique():
                cond = (data_lagged_features[id_] == sku)
                data_lagged_features.loc[cond, c] = data_lagged_features[cond][c].fillna(data_lagged_features[cond][c].mean())
        if data_lagged_features[data_lagged_features.drop([date_col, target], axis=1).columns].isna().sum().sum() == 0:
            break
    
    data_lagged_features = data_lagged_features.set_index([date_col] + id_cols)
    df_train = data_lagged_features[data_lagged_features[target].notna()]
    df_test = data_lagged_features[data_lagged_features[target].isna()]
    target_mean, target_stdev = scale(df_train, df_test, target, scl)
    features = df_train.drop([target], axis=1).columns
    set_random_seed(7)
    return df_train, df_test, target_mean, target_stdev, features


def train_one_epoch(data_loader, model, loss_function, optimizer, scheduler, eval_=True):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()
    
    for X, y in tqdm_notebook(data_loader):
        output = model(X)
        loss = loss_function(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()

    avg_loss = total_loss / num_batches
    if eval_:
        print(f"Train loss: {avg_loss}")


def predict(data_loader, model):
    output = torch.tensor([])
    model.eval()
    with torch.no_grad():
        for X, _ in tqdm_notebook(data_loader):
            y_star = model(X)
            output = torch.cat((output, y_star), 0)

    return output


def smape(a, f):
    return 1 / len(a) * np.sum(2 * np.abs(f - a) / (np.abs(a) + np.abs(f)) * 100)


def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_df_out(model, df_train, df_test, train_eval_loader, test_loader, target, target_mean, target_stdev):
    ystar_col = "Model forecast"
    df_train[ystar_col] = predict(train_eval_loader, model).numpy()
    df_test[ystar_col] = predict(test_loader, model).numpy()
    df_out = pd.concat((df_train, df_test))[[target, ystar_col]]
    for c in df_out.columns:
        df_out[c] = df_out[c] * target_stdev + target_mean

    return df_out

def get_submission(path, df_out, target, id_cols, date_col):
    test = pd.read_csv(path)
    test[date_col] = pd.to_datetime(test[date_col], dayfirst=True)
    df_out = df_out.sort_index()
    ans = df_out[df_out[target].isna()]
    ans = ans.drop(target, axis=1)
    ans.rename(columns={'Model forecast': 'Forecast'}, inplace=True)
    ans = ans.reset_index()
    ans[date_col] = pd.to_datetime(ans[date_col])
    ans = ans.set_index([date_col] + id_cols)
    ans['Forecast'] = pd.merge(test.reset_index(), ans.reset_index(), on=([date_col] + id_cols), how='left')['Forecast'].values
    ans.index = [np.arange(5970)]
    ans.rename(columns={'Forecast' : target}, inplace=True)
    ans.to_csv('ans.csv', index_label='id')
    return ans

def train(df_train, epochs, train_loader, train_eval_loader,model,
      loss_function, optimizer, scheduler, target_mean, target_stdev, eval_=True):
    for ix_epoch in range(epochs):
        print(f"Epoch {ix_epoch}\n---------")
        train_one_epoch(train_loader, model, loss_function, optimizer=optimizer, scheduler=scheduler, eval_=eval_)
        if eval_:
            target='Demand'
            ystar_col = "Model forecast"
            df_train[ystar_col] = predict(train_eval_loader, model).numpy()

            df_out = df_train[[target, ystar_col]]

            for c in df_out.columns:
                df_out[c] = df_out[c] * target_stdev + target_mean

            smape_ = smape(df_out.dropna()['Demand'], df_out.dropna()['Model forecast'])
            print('Smape =', smape_)

            print()
            
            
def get_dataloaders(df_train, df_test, target, features, batch_size=128, sequence_length=30):
    train_dataset = SequenceDataset(
        df_train,
        target=target,
        features=features,
        sequence_length=sequence_length
    )
    test_dataset = SequenceDataset(
        df_test,
        target=target,
        features=features,
        sequence_length=sequence_length
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
    train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, train_eval_loader

