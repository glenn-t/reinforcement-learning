test_agents = function(N, p1, p2, table = TRUE) {
  winner = replicate(N, {
    play_game(list(p1, p2), draw = FALSE, all_states)$winner
  })
  if(!table) {
    return(winner)
  } else {
    return(table(winner))
  }
}