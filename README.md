# DLDF
Deep learning for Demand Forecasting

# Features
## Params Dictionary
 |  #  | Column                            | Dtype   | Description
 |:--- |:----------------------------------|:--------|:----------------------------------------------------------------------------------
 | 0   | data                              | pd.DataFrame  | Данные, для которых осуществляется генерация признаков
 | 1   | target_cols                       | object  | Name of the tournament the recorded match belongs to.
 | 2   | id_cols                           | object  | Court construction type - surface material of the court the match is played on.
 | 3   | date_col                          | int64   | Number of players in a tournament rounded to nearest power of 2.
 | 4   | lags                              | object  | Level of tournament: G = Grand Slams, M = Masters 1000, A = Other tour level events.
 | 5   | windows                           | int64   | Match specific identifier within the tourney id.
 | 6   | preagg_methods                    | object  | The final results of the match outcome.
 | 7   | agg_methods                       | int64   | The match format. 3 = Best of 3 sets, 5 = Best of 5 sets for the match.
 | 8   | dynamic_filters                   | object  | What round the match is in a tournament. RR = Round robin, ER = Early rounds.
 | 9   | ewm_params                        | float64 | Match length.

