# Causes the agent to run it's own choose action function and return the result

choose_action = function(agent, board) {
  UseMethod("choose_action")
}

choose_action.agent = function(agent, board) {
  agent$choose_action(agent, board)
}

choose_action.default = function(agent, board) {
  stop("Must be an 'agent' class")
}