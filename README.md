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
 | 11  | player_2                          |  50470 non-null | object  | The other player featured in the match.
 | 12  | player_1_age                      |  50470 non-null | float64 | Age of player 1 at the time of the match.
 | 13  | player_2_age                      |  50470 non-null | float64 | Age of player 2 at the time of the match.
 | 14  | player_1_hand                     |  50470 non-null | object  | Dominant hand for player 1.
 | 15  | player_2_hand                     |  50470 non-null | object  | Dominant hand for player 2.
 | 16  | player_1_ht                       |  50470 non-null | float64 | Height of player 1.
 | 17  | player_2_ht                       |  50470 non-null | float64 | Height of player 2.
 | 18  | player_1_id                       |  50470 non-null | int64   | Unique player identifier for player 1.
 | 19  | player_2_id                       |  50470 non-null | int64   | Unique player identifier for player 2.
 | 20  | player_1_ioc                      |  50470 non-null | object  | Country of origin for player 1.
 | 21  | player_2_ioc                      |  50470 non-null | object  | Country of origin for player 2.
 | 22  | player_1_rank                     |  50470 non-null | float64 | Player 1 rank at the start of the tournament.
 | 23  | player_2_rank                     |  50470 non-null | float64 | Player 2 rank at the start of the tournament.
 | 24  | player_1_rank_points              |  50470 non-null | float64 | Player 1 rank points at the start of the tournament.
 | 25  | player_2_rank_points              |  50470 non-null | float64 | Player 2 rank points at the start of the tournament.
 | 26  | player_1_seed                     |  13218 non-null | float64 | Player 1 seed for the tournament, if seeded.
 | 27  | player_2_seed                     |  14181 non-null | float64 | Player 2 seed for the tournament, if seeded.
 | 28  | player_1_aces                     |  50470 non-null | float64 | Number of serves from player 1 in the match completely untouched by player 2.
 | 29  | player_2_aces                     |  50470 non-null | float64 | Number of serves from player 2 in the match completely untouched by player 1.
 | 30  | player_1_double_faults            |  50470 non-null | float64 | Number of times player 1 failed to start a point by faulting twice (free p2 point).
 | 31  | player_2_double_faults            |  50470 non-null | float64 | Number of times player 2 failed to start a point by faulting twice (free p1 point).
 | 32  | player_1_service_points           |  50470 non-null | float64 | Number of points player 1 played on his serve.
 | 33  | player_2_service_points           |  50470 non-null | float64 | Number of points player 2 played on his serve.
 | 34  | player_1_first_serves_in          |  50470 non-null | float64 | Number of first serves player 1 made.
 | 35  | player_2_first_serves_in          |  50470 non-null | float64 | Number of first serves player 2 made.
 | 36  | player_1_first_serve_points_won   |  50470 non-null | float64 | Number of first serve points won by player 1.
 | 37  | player_2_first_serve_points_won   |  50470 non-null | float64 | Number of first serve points won by player 2.
 | 38  | player_1_second_serve_points_won  |  50470 non-null | float64 | Number of second serve points won by player 1.
 | 39  | player_2_second_serve_points_won  |  50470 non-null | float64 | Number of second serve points won by player 2.
 | 40  | player_1_service_game_total       |  50470 non-null | float64 | Number of games player 1 served in a match.
 | 41  | player_2_service_game_total       |  50470 non-null | float64 | Number of games player 2 served in a match.
 | 42  | player_1_break_points_saved       |  50470 non-null | float64 | Number of points player 1 won to stave off a break of serve.
 | 43  | player_2_break_points_saved       |  50470 non-null | float64 | Number of points player 2 won to stave off a break of serve.
 | 44  | player_1_break_points_faced       |  50470 non-null | float64 | Number of break points player 1 faced.
 | 45  | player_2_break_points_faced       |  50470 non-null | float64 | Number of break points player 2 faced.
 | 46  | winner                            |  50470 non-null | object  | The name of the winner.
 | 47  | player_1_first_serve_%            |  50470 non-null | float64 | Percent of first serves in for player 1.
 | 48  | player_2_first_serve_%            |  50470 non-null | float64 | Percent of first serves in for player 2.
 | 49  | player_1_first_serve_win_%        |  50470 non-null | float64 | Percent of first service points won for player 1.
 | 50  | player_2_first_serve_win_%        |  50470 non-null | float64 | Percent of first service points won for player 2.
 | 51  | player_1_break_points_won         |  50470 non-null | float64 | Number of times player 1 broke player 2's service.
 | 52  | player_2_break_points_won         |  50470 non-null | float64 | Number of times player 2 broke player 2's service.
 | 53  | player_2_seeded                   |  50470 non-null | bool    | Boolean value that designates whether or not player 2 is seeded.
 | 54  | player_1_seeded                   |  50470 non-null | bool    | Boolean value that designates whether or not player 1 is seeded.
 | 55  | surface_Carpet                    |  50470 non-null | uint8   | Whether or not the match was played on carpet. 1 = Yes, 0 = No.
 | 56  | surface_Clay                      |  50470 non-null | uint8   | Whether or not the match was played on clay. 1 = Yes, 0 = No.
 | 57  | surface_Grass                     |  50470 non-null | uint8   | Whether or not the match was played on grass. 1 = Yes, 0 = No.
 | 58  | surface_Hard                      |  50470 non-null | uint8   | Whether or not the match was played on hard court. 1 = Yes, 0 = No.
 | 59  | tourney_level_A                   |  50470 non-null | uint8   | Whether or not the tournament was an tour level event. 1 = Yes, 0 = No.
 | 60  | tourney_level_D                   |  50470 non-null | uint8   | Whether or not the tournament was a Davis Cup event. 1 = Yes, 0 = No.
 | 61  | tourney_level_F                   |  50470 non-null | uint8   | Whether or not the tournament was a Tour Finals or season-ending event. 1 = Yes, 0 = No.
 | 62  | tourney_level_G                   |  50470 non-null | uint8   | Whether or not the tournament was a Grand Slam event. 1 = Yes, 0 = No.
 | 63  | tourney_level_M                   |  50470 non-null | uint8   | Whether or not the tournament was a Masters 1000 event. 1 = Yes, 0 = No.
 | 64  | player_1_hand_L                   |  50470 non-null | uint8   | Whether or not player 1 plays left-handed. 1 = Yes, 0 = No.
 | 65  | player_1_hand_R                   |  50470 non-null | uint8   | Whether or not player 1 plays right-handed. 1 = Yes, 0 = No.
 | 66  | player_2_hand_L                   |  50470 non-null | uint8   | Whether or not player 2 plays left-handed. 1 = Yes, 0 = No.
 | 67  | player_2_hand_R                   |  50470 non-null | uint8   | Whether or not player 2 plays right-handed. 1 = Yes, 0 = No.
 | 68  | round_ER                          |  50470 non-null | uint8   | Whether or not the match was in the early rounds. 1 = Yes, 0 = No.
 | 69  | round_BR                          |  50470 non-null | uint8   | W.o.n.t.m.w a (bronze round) in 2016 Olympics/ 2018 season-ending event. 1 = Y, 0 = N
 | 70  | round_F                           |  50470 non-null | uint8   | Whether or not the match was the final round. 1 = Yes, 0 = No.
 | 71  | round_QF                          |  50470 non-null | uint8   | Whether or not the match was a quarter-final. 1 = Yes, 0 = No.
 | 72  | round_R128                        |  50470 non-null | uint8   | Whether or not the match was in the round of 128 1 = Yes, 0 = No.
 | 73  | round_R16                         |  50470 non-null | uint8   | Whether or not the match was in the round of 16. 1 = Yes, 0 = No.
 | 74  | round_R32                         |  50470 non-null | uint8   | Whether or not the match was in the round of 32. 1 = Yes, 0 = No.
 | 75  | round_R64                         |  50470 non-null | uint8   | Whether or not the match was in the round of 64. 1 = Yes, 0 = No.
 | 76  | round_RR                          |  50470 non-null | uint8   | Whether or not the match was a round robin match. 1 = Yes, 0 = No.
 | 77  | round_SF                          |  50470 non-null | uint8   | Whether or not the match was a semifinal. 1 = Yes, 0 = No.
 | 78  | h2h_1                             |  50470 non-null | int64   | Number that represents how many victories player1 has over player 2.
 | 79  | h2h_1                             |  50470 non-null | int64   | Number that represents how many victories player2 has over player 1.
 | 80  | player_1_wins                     |  50470 non-null | bool    | Target variable. Boolean value that designates whether or not player 1 won the match.
