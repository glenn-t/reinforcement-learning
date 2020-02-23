# # Creates a intelligent agent
# new_agent_01 = function(all_states, symbol) {
#   list(
#     "value_function" = {
#       # Initialise value function
#       all_states$value = 0.5
#       all_states$value[all_states$winner == ]
#       # return all states
#       all_states
#     }
#     "choose_action" = function(self, board) {
#       available_actions = which(board == 0)
#       action_ind = sample(length(available_actions), size = 1)
#       return(available_actions[action_ind])
#     }
#   )
# }