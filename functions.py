import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import timedelta
from scipy import stats
from statsmodels.tsa import stattools
from copy import deepcopy
import neptune.new as neptune
import seaborn as sns
import warnings, pylab
from tqdm import tqdm
import random
import torch

warnings.filterwarnings('ignore')
sns.set_style("darkgrid")
pd.plotting.register_matplotlib_converters()
sns.mpl.rc("figure", figsize=(25, 5))
sns.mpl.rc("font", size=14)


def scale(df_train, df_test, target):
    target_mean = df_train[target].mean()
    target_stdev = df_train[target].std()
    for c in df_train.columns:
        mean = df_train[c].mean()
        stdev = df_train[c].std()
        df_train[c] = (df_train[c] - mean) / stdev
        df_test[c] = (df_test[c] - mean) / stdev
    return target_mean, target_stdev

def preprocess(data_lagged_features):
    data_lagged_features = data_lagged_features.set_index(['Date', 'Store_id', 'SKU_id'])
    df_train = data_lagged_features[data_lagged_features['Demand'].notna()]
    df_test = data_lagged_features[data_lagged_features['Demand'].isna()]
    df_train = df_train.drop('Promo_Price', axis=1).dropna()
    df_test = df_test.drop('Promo_Price', axis=1)
    target = 'Demand'
    target_mean, target_stdev = scale(df_train, df_test, target)
    features = df_train.drop(['Demand', 'Promo'], axis=1).columns
    return df_train, df_test, target_mean, target_stdev, target, features


def train_one_epoch(data_loader, model, loss_function, optimizer, scheduler):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()

    for X, y in tqdm(data_loader):
        output = model(X)
        loss = loss_function(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()

    avg_loss = total_loss / num_batches
    print(f"Train loss: {avg_loss}")


def predict(data_loader, model):
    output = torch.tensor([])
    model.eval()
    with torch.no_grad():
        for X, _ in tqdm(data_loader):
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

def get_submission(path, df_out):
    test = pd.read_csv(path)
    test['Date'] = test['Date'].str.split('.').apply(lambda x : '-'.join(x[::-1]))
    test['Date'] = pd.to_datetime(test['Date'])
    df_out = df_out.sort_index()
    ans = df_out[df_out['Demand'].isna()]
    ans = ans.drop('Demand', axis=1)
    ans.rename(columns={'Model forecast': 'Forecast'}, inplace=True)
    ans = ans.reset_index()
    ans['Date'] = pd.to_datetime(ans['Date'])
    ans = ans.set_index(['Date', 'Store_id', 'SKU_id'])
    ans['Forecast'] = pd.merge(test.reset_index(), ans.reset_index(), on=['Date', 'Store_id', 'SKU_id'], how='left')['Forecast'].values
    ans.index = [np.arange(5970)]
    ans.rename(columns={'Forecast' : 'Demand'}, inplace=True)
    ans.to_csv('try_one.csv', index_label='id')
    return ans

