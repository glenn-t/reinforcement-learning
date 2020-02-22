# Creates a human agent
new_human = function() {
  list(
    "choose_action" = function(self, board) {
      available_actions = which(board == 0)
      valid_result = FALSE
      while(!valid_result) {
        action = readline("Select position (1-9): ") %>%
          as.integer()
          if(action %in% available_actions) {
            valid_result = TRUE
          } else {
            cat("\nPlease enter a valid action")
          }
        }
      return(action)
    }
  )
}