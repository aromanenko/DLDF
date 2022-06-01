import pandas as pd


def process(data):
	data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
	data.set_index(['Date'], inplace=True)
	data['Promo'] = data['Promo'].fillna(0)
	data.reset_index().set_index(['SKU_id', 'Store_id', 'Date'], inplace=True)
	data['Regular_Price'] = data['Regular_Price'].ffill().bfill()
	data.reset_index().set_index(['Date'], inplace=True)

	# add actual price (promo price when promo occurred or regular price otherwise)
	data['Actual_Price'] = data['Promo_Price'].combine_first(data['Regular_Price'])
	data['Promo_percent'] = (1 - (data['Actual_Price'] / data['Regular_Price']))
	data = data.drop('Promo_Price', axis=1)

	data.reset_index(inplace=True)
	data["weekday"] = data['Date'].dt.weekday
	# data['weekday_name'] = data['Date'].dt.strftime('%A')
	data["monthday"] = data['Date'].dt.day
	data['is_weekend'] = data['weekday'].isin([5, 6]) * 1
	data['month_period'] = 0
	data.loc[data['monthday'] >= 15, 'month_period'] = 1

	## base feature

	data['demand_expanding_mean'] = data.groupby(['Store_id', 'SKU_id'])['Demand'].expanding().mean().droplevel(['Store_id', 'SKU_id'])
	data['demand_expanding_mean'] = data.groupby(['Store_id', 'SKU_id'])['demand_expanding_mean'].apply(lambda x: x.shift(14))


	return data
