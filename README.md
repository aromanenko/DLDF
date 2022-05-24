# DLDF
Deep learning for Demand Forecasting

# Features
## Params Dictionary
 |  #  | Column                            |  Non-Null Count | Dtype   | Description
 |:--- |:----------------------------------|:--------------- |:--------|:----------------------------------------------------------------------------------
 | 0   | tourney_id                        |  50470 non-null | object  | Unique identifier for the tournament that the record of match data belongs to.
 | 1   | tourney_name                      |  50470 non-null | object  | Name of the tournament the recorded match belongs to.
 | 2   | surface                           |  50470 non-null | object  | Court construction type - surface material of the court the match is played on.
 | 3   | draw_size                         |  50470 non-null | int64   | Number of players in a tournament rounded to nearest power of 2.
 | 4   | tourney_level                     |  50470 non-null | object  | Level of tournament: G = Grand Slams, M = Masters 1000, A = Other tour level events.
 | 5   | match_num                         |  50470 non-null | int64   | Match specific identifier within the tourney id.
 | 6   | score                             |  50470 non-null | object  | The final results of the match outcome.
 | 7   | best_of                           |  50470 non-null | int64   | The match format. 3 = Best of 3 sets, 5 = Best of 5 sets for the match.
 | 8   | round                             |  50470 non-null | object  | What round the match is in a tournament. RR = Round robin, ER = Early rounds.
 | 9   | minutes                           |  35543 non-null | float64 | Match length.
 | 10  | player_1                          |  50470 non-null | object  | One of the players featured in a match.

