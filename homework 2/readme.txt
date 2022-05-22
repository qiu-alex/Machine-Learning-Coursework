1) Alex Qiu

2) Bot Marius

3) The evaluation function keeps count of all possible streaks in addition
to all streaks present in the state on each state. I think it is an effective
estimate of the utility of each state since its scoring is based on how many 
streaks each playe already has and how many possible streaks there are left in 
the board

4) The test board was a simple 3 x 3 board to test that the minimax function
will always prioritize putting a piece that will create a streak
