# Training functons
train_agents = function(N, p1, p2, draw = FALSE, print = TRUE) {
  winner = integer(length = N)
  for(i in 1:N) {
    res = play_game(list(p1, p2), draw = draw, all_states)
    p1 = update_agent(p1, res$state_history)
    p2 = update_agent(p2, res$state_history)
    res$winner[res$winner == 2] = -1
    winner[i] = res$winner
    if((i %% 200 == 0) & print) {
      ind = (i-199):i
      cat(i, sum(winner[ind] == 0), sum(winner[ind] == 1), sum(winner[ind] == -1),mean(winner[1:i]),  "\n")
    }
  }
  return(list("p1" = p1, "p2" = p2, winner = winner))
}