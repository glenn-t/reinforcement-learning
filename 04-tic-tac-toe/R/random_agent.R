# Creates a random agent
new_agent_random = function() {
  
  out = list(
    "choose_action" = function(self, board) {
      available_actions = which(board == 0)
      action_ind = sample(length(available_actions), size = 1)
      return(available_actions[action_ind])
    }
  )
  
  class(out) = "agent"
  return(out)
}