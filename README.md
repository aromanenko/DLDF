# DLDF
Deep learning for Demand Forecasting

# Features
## Params Dictionary
 |  #  | Column                            | Dtype   | Description
 |:--- |:----------------------------------|:--------|:----------------------------------------------------------------------------------
 | 0   | data                              | pd.DataFrame  | Данные, для которых осуществляется генерация признаков
 | 1   | target_cols                       | list    | Name of the tournament the recorded match belongs to.
 | 2   | id_cols                           | list    | Court construction type - surface material of the court the match is played on.
 | 3   | date_col                          | str     | Number of players in a tournament rounded to nearest power of 2.
 | 4   | lags                              | list    | Level of tournament: G = Grand Slams, M = Masters 1000, A = Other tour level events.
 | 5   | windows                           | list    | Match specific identifier within the tourney id.
 | 6   | preagg_methods                    | list    | The final results of the match outcome.
 | 7   | agg_methods                       | list    | The match format. 3 = Best of 3 sets, 5 = Best of 5 sets for the match.
 | 8   | dynamic_filters                   | list    | What round the match is in a tournament. RR = Round robin, ER = Early rounds.
 | 9   | ewm_params                        | dict    | Match length.

