import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from datetime import date
from datetime import datetime
import math

def fix_nan(csv_data):
    """

    :param csv_data: pd.DataFrame from test.csv
    :return: DataFrame with fixed nan in some columns and make data as datetime
    """
    csv_data["period_start_dt"] = pd.to_datetime(csv_data["period_start_dt"], format='%Y.%m.%d')
    bad_column = ["PROMO1_FLAG","PROMO2_FLAG","NUM_CONSULTANT","AUTORIZATION_FLAG"]
    for replace_name in bad_column:
        csv_data[replace_name] = csv_data[replace_name].fillna(0.0)
    csv_data = csv_data.drop(columns=['PROMO2_FLAG', 'NUM_CONSULTANT'])
    return csv_data
    
def percentile(n):
    '''Calculate n - percentile of data'''
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'pctl%s' % n
    return percentile_

def lagged_features(df
                    , lags = [7, 14, 21, 28]
                    , windows = [7, 14]
                    , aggregation_methods = {'mean', 'median', percentile(10),  percentile(90)}
                    , promo_filters = [0, 1]
                    , target_var = 'demand'
                    , by_all_stores = False
                    , by_all_products = False):
    
    if len(df) == 0:
        return df
    
                
    
    # loop by filter variables and window
    for w in windows:

        # check whether filtered df in not empty
        if len(df) > 0:
            
            # lagged features calculation
            lf_df = df.set_index(['product_rk', 'store_location_rk', 'period_start_dt'])\
                 [target_var].groupby(level=['product_rk','store_location_rk']).\
                apply(lambda x: x.rolling(window=w, min_periods=1).agg(aggregation_methods))

            # provide lags tranformations
            for l in lags:
                new_names = {x: "{3}_lag{0}_wdw{1}_{2}".
                              format(l, w, x, target_var) for x in lf_df.columns }

                df = df.merge(lf_df.shift(l).reset_index().rename(columns = new_names),
                    how='left', on =['product_rk', 'store_location_rk', 'period_start_dt'] )
                
                
            if by_all_stores:
                lf_df = df.set_index(['product_rk', 'period_start_dt'])\
                    [target_var].groupby(level=['product_rk', 'period_start_dt']).median().\
                    groupby(level=['product_rk']).\
                    apply(lambda x: x.rolling(window=w, min_periods=1).agg(aggregation_methods))

                for l in lags:
                    new_names = {x: "all_stores_{3}_lag{0}_wdw{1}_{2}".
                                  format(l, w, x, target_var) for x in lf_df.columns }

                    df = df.merge(lf_df.shift(l).reset_index().rename(columns = new_names),
                        how='left', on =['product_rk', 'period_start_dt'] )
                    
            if by_all_products:
                lf_df = df.set_index(['store_location_rk', 'period_start_dt'])[target_var]\
                .groupby(level=['store_location_rk', 'period_start_dt']).median().\
                    groupby(level=['store_location_rk']).\
                    apply(lambda x: x.rolling(window=w, min_periods=1).agg(aggregation_methods))

                for l in lags:
                    new_names = {x: "all_products_{3}_lag{0}_wdw{1}_{2}".
                                  format(l, w, x, target_var) for x in lf_df.columns }

                    df = df.merge(lf_df.shift(l).reset_index().rename(columns = new_names),
                        how='left', on =['store_location_rk', 'period_start_dt'] )

    return df
    
def add_holiday_flag(df, holiday, flag_name):
    holiay_dt = datetime.strptime(holiday, '%Y-%m-%d')
    
    min_year = df['period_start_dt'].min().year
    max_year = df['period_end_dt'].max().year
    
    for year in range(min_year, max_year + 1):
        dt = datetime(year, holiay_dt.month, holiay_dt.day)
        mask = (df['period_start_dt'] < dt) & (df['period_end_dt'] >= dt)
        df.loc[mask, flag_name] = 1
        df[flag_name] = df[flag_name].fillna(0)
        df[flag_name] = df[flag_name].astype(int)
        
    return df
    
def add_all_holidays(final):
    final["no_year"] = final['period_start_dt'].map(lambda x: x.strftime("%m-%d"))
    final["period_end_dt"] = final['period_start_dt'] + pd.DateOffset(days=6)

    final = add_holiday_flag(final, "2013-02-14", "flag_14FEB")
    final = add_holiday_flag(final, "2013-02-23", "flag_23FEB")
    final = add_holiday_flag(final, "2013-03-08", "flag_8MAR")
    final = add_holiday_flag(final, "2013-05-09", "flag_9MAY")
    final = add_holiday_flag(final, "2013-09-01", "flag_1SEP")
    final = add_holiday_flag(final, "2013-01-01", "flag_NEW_YEAR")

    final = final.drop(columns=['period_end_dt'])
    final = final.drop(columns=['no_year'])
    return final
    
def TheilWageSmoothing(x, params, h=1):
    """
    Get predictions for timeseries using Theil-Wage model
    
    :x - time series
    :params - dict with keys: "alpha", "beta", "gamma", "season", "damp_trend"
    :h - horison
    
    Return
    :y_pred - predictions, size n + h (n - size of time series)
    :l - forecast parameter without trend and seasonality, size n
    :b - trend component, size n
    :s - seasonal component, size n
    """
    T = len(x)
    x.index = np.arange(T)
    alpha = params['alpha']
    beta = params['beta']
    gamma = params['gamma']
    p = params['season']
    damp_trend = params['damp_trend']
    
    y_pred = [np.NaN]*(T+h)
    
    l = [np.NaN] * T
    l[0] = x[0]
    b = [np.NaN] * T
    b[0] = 0
    s = [np.NaN] * (T + p)
    for i in range(p):
        if i < T:
            s[i] = x[i]
        else:
            s[i] = s[i-1]
            
    sum_damp_trend = 0
    for i in range(1, h + 1):
        sum_damp_trend += damp_trend ** i

    for i in range(T):
        if not math.isnan(x[i]):
            if i < p and i > 0:
                l[i] = alpha * (x[i] - s[i]) + (1 - alpha) * (l[i-1] + b[i-1] * damp_trend)
                b[i] = beta * (l[i] - l[i-1]) + (1 - beta) * b[i-1] * damp_trend
                y_pred[i+h] = l[i] + sum_damp_trend * b[i] + s[i]
            elif i >= p and i > 0:
                s_old = s[i - p]
                l[i] = alpha * (x[i] - s_old) + (1 - alpha) * (l[i-1] + b[i-1] * damp_trend)
                b[i] = beta * (l[i] - l[i-1]) + (1 - beta) * b[i-1] * damp_trend
                s[i] = gamma * (x[i] - l[i]) + (1 - gamma) * s[i - p]
                y_pred[i+h] = l[i] +  sum_damp_trend * b[i] + s[i + h%p - p]
            if y_pred[i+h] < 0:
                y_pred[i+h] = 0
                
    return y_pred, l[:T], b[:T], s[:T]
    
def HoltWintersSmoothing(x, params, h=1):
    """
    Get predictions for timeseries using Theil-Wage model
    
    :x - time series
    :params - dict with keys: "alpha", "beta", "gamma", "season"
    :h - horison
    
    Return
    :y_pred - predictions, size n + h (n - size of time series)
    :l - forecast parameter without trend and seasonality, size n
    :b - trend component, size n
    :s - seasonal component, size n
    """
    T = len(x)
    x.index = np.arange(T)
    alpha = params['alpha']
    beta = params['beta']
    gamma = params['gamma']
    p = params['season']
    
    y_pred = [np.NaN]*(T+h)
    
    l = [np.NaN] * T
    l[0] = x[0]
    b = [np.NaN] * T
    b[0] = 0
    s = [np.NaN] * (T + p)
    for i in range(p):
        if i < T:
            s[i] = x[i]
        else:
            s[i] = s[i-1]

    for i in range(T):
        if not math.isnan(x[i]):
            if i < p and i > 0:
                l[i] = alpha * (x[i] / s[i]) + (1 - alpha) * (l[i-1] * b[i-1])
                b[i] = beta * (l[i] / l[i-1]) + (1 - beta) * b[i-1]
                y_pred[i+h] = l[i] + b[i] + s[i]
            elif i >= p and i > 0:
                s_old = s[i - p]
                l[i] = alpha * (x[i] / s_old) + (1 - alpha) * (l[i-1] * b[i-1])
                b[i] = beta * (l[i] / l[i-1]) + (1 - beta) * b[i-1]
                s[i] = gamma * (x[i] / l[i]) + (1 - gamma) * s[i - p]
                y_pred[i+h] = l[i] * b[i] * s[i + h%p - p]
            if y_pred[i+h] < 0:
                y_pred[i+h] = 0
                
    return y_pred, l[:T], b[:T], s[:T]
    
def add_HW_TW(data):
    data['l_tw'] = 0
    data['s_tw'] = 1
    data['b_tw'] = 0
    data['y_tw'] = 0

    data['l_hw'] = 0
    data['s_hw'] = 1
    data['b_hw'] = 0
    data['y_hw'] = 0

    for prod in data.product_rk.unique():
        for store in data.store_location_rk.unique():
            X_curr = data[(data.product_rk == prod) & (data.store_location_rk == store) & ~data.demand.isna()]
            if X_curr.shape[0] == 0:
                continue
            params = {"alpha": 0.2, "beta": 0.01, "gamma": 0.5, "season": 52, "damp_trend": 0.9}
            h = 42
            n = X_curr.demand.shape[0]
            y_hat, l, b, s = TheilWageSmoothing(X_curr.demand, params, h=h)
            data.loc[(data.product_rk == prod) & (data.store_location_rk == store) & ~data.demand.isna(), 'y_tw'] = y_hat[:-h]
            data.loc[(data.product_rk == prod) & (data.store_location_rk == store) & ~data.demand.isna(), 'l_tw'] = l
            data.loc[(data.product_rk == prod) & (data.store_location_rk == store) & ~data.demand.isna(), 'b_tw'] = b
            data.loc[(data.product_rk == prod) & (data.store_location_rk == store) & ~data.demand.isna(), 's_tw'] = s
            
            params = {"alpha": 0.3, "beta": 0.01, "gamma": 0.5, "season": 52}
            h = 42
            n = X_curr.demand.shape[0]
            y_hat, l, b, s = HoltWintersSmoothing(X_curr.demand, params, h=h)
            data.loc[(data.product_rk == prod) & (data.store_location_rk == store) & ~data.demand.isna(), 'y_hw'] = y_hat[:-h]
            data.loc[(data.product_rk == prod) & (data.store_location_rk == store) & ~data.demand.isna(), 'l_hw'] = l
            data.loc[(data.product_rk == prod) & (data.store_location_rk == store) & ~data.demand.isna(), 'b_hw'] = b
            data.loc[(data.product_rk == prod) & (data.store_location_rk == store) & ~data.demand.isna(), 's_hw'] = s
    return data

def process(csv_data, analytic_methods=False):
    """

    :param csv_data: pd.DataFrame from test.csv
    :return: fixed DataFrame with one-hot for product and store
    """
    csv_data = fix_nan(csv_data)
    
    def make_column_one_hot(data, column_name):
        data = pd.merge(data, pd.get_dummies(data[column_name], prefix=column_name), left_index=True, right_index=True)
        return data
#        return data.drop(columns=column_name)

    def prepare_data(data):
        data = lagged_features(data)
        if analytic_methods:
            data = add_HW_TW(data)
        data = make_column_one_hot(data,"product_rk")
        data = make_column_one_hot(data,"store_location_rk")
        data = add_all_holidays(data)
        return data
        
    return prepare_data(csv_data)

def SMAPE(predicted,target):
    target = np.array(target)
    def SMAPE_base(one, target):
        one_np = np.array(one)
        return np.sum(np.abs(one_np-target)/(2*one_np+2*target))/one_np.size
    if len(predicted)==0:
        return []
    if isinstance(predicted,list):
        if isinstance(predicted[0],list):
            return [SMAPE_base(e,target) for e in predicted]
        else:
            return [SMAPE_base(predicted,target)]
    else:
        if predicted.ndim==2:
            return np.sum(np.abs(predicted-target)/(2*predicted+2*target),axis=0)/predicted.shape[1]
        else:
            return [SMAPE_base(predicted,target)]
def WAPE(predicted,target):
    def SMAPE_base(one, target):
        target = np.array(target)
        one_np = np.array(one)
        return np.sum(np.abs(one_np-target))/np.sum(target)
    if len(predicted)==0:
        return []
    if isinstance(predicted,list):
        if isinstance(predicted[0],list):
            return [SMAPE_base(e,target) for e in predicted]
        else:
            return [SMAPE_base(predicted,target)]
    else:
        if predicted.ndim==2:
            return np.sum(np.abs(predicted-target),axis=0)/np.sum(predicted,axis=0)
        else:
            return [SMAPE_base(predicted,target)]
        


def plot(data, model):
    """
    ploting total demand overall by all shops and products from 2019-12 to 2020-01 using trained model by data from 2017-01 to 2019-11
    :params
        data: processed pd.DataFrame with target demand and features
        model: trained model
    """
    from matplotlib import pyplot as plt
    #выборка пар магазин-товар, данные спроса по которым не содержат пропуска
    unique_items_grouped = data.groupby(["product_rk","store_location_rk"],as_index=False)
    unique_items_sizes = unique_items_grouped.count()[["product_rk","store_location_rk","period_start_dt"]]
    unique_items_sizes = pd.DataFrame(unique_items_sizes).rename(columns={"period_start_dt":"days in table"})

    full_data_pairs = unique_items_sizes[unique_items_sizes["days in table"]==unique_items_sizes["days in table"].max()]
    full_data_pairs = full_data_pairs.reset_index()[["product_rk","store_location_rk"]]
    
    
    def get_sum_demand_by_product(full_data_pairs, number):
        """
        aggregating by number(product_id) and getting sum of demand
        :params
            full_data_pairs: data with prroduct_rk and storre_location_rk
            number: product_id
        :return pd.DataFrame period_start_id and sum(demand)
        """
        print(number)
        print(full_data_pairs['product_rk'].unique().shape[0])
        if number < 0 or number>=full_data_pairs['product_rk'].unique().shape[0]:
            raise Exception("numer out of array ( from 0 to {})".format(full_data_pairs['product_rk'].unique().shape[0]))
        cur_product = full_data_pairs['product_rk'].unique()[number]
        shop_list = full_data_pairs[full_data_pairs['product_rk']==cur_product]['store_location_rk']
        cut_datalist = data[(data['store_location_rk'].isin(shop_list))&(data['product_rk']==cur_product)]
        return cut_datalist.groupby(['period_start_dt']).sum()[['demand']].reset_index()
    
    def get_sum_demand_by_store(full_data_pairs, number):
        """
        aggregating by number(store_location_id) and getting sum of demand
        :params
            full_data_pairs: data with product_rk and storre_location_rk
            number: store_id
        :return pd.DataFrame period_start_id and sum(demand)
        """
        print(number)
        print(full_data_pairs['store_location_rk'].unique().shape[0])
        if number < 0 or number >= full_data_pairs['store_location_rk'].unique().shape[0]:
            raise Exception("numer out of array ( from 0 to {})".format(full_data_pairs['store_location_rk'].unique().shape[0]))
        cur_store = full_data_pairs['store_location_rk'].unique()[number]
        store_list = full_data_pairs[full_data_pairs['product_rk']==cur_product]['product_rk']
        cut_datalist = data[(data['product_rk'].isin(shop_list))&(data['product_rk']==cur_product)]
        return cut_datalist.groupby(['period_start_dt']).sum()[['demand']].reset_index()
    
    # ploting sum(demand) by product_rk
    matplotlib.pyplot.plot([1,2], [1,2])
    for index, number in enumerate(full_data_pairs['product_rk'].unique()):
        product_data = get_sum_demand_by_product(full_data_pairs, index)
        plt.figure(figsize=(10, 8))
        plt.plot(product_data["days in table"], product_data.demand)
        plt.title("total demand by product id")
        plt.xlabel("period_start_id")
        plt.ylabel("demand")
        plt.show()
    
    for index, number in enumerate(full_data_pairs['store_location_rk'].unique()):
        store_data = get_sum_demand_by_store(full_data_pairs, index)
        plt.figure(figsize=(10, 8))
        plt.plot(store_data["days in table"], store_data.demand)
        plt.title("total demand by sore id")
        plt.xlabel("period_start_id")
        plt.ylabel("demand")
        plt.show()
    
    # plt.figure(figsize=(10, 8))
    # pred = model(data)
    # plt.plot(data.period_start_dt, pred)
    # plt.plot(data.period_start_dt, data.demand)
    # plt.show()
