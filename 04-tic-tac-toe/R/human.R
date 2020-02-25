# Creates a human agent
new_human = function() {
  
  out = list(
    "choose_action" = function(self, board) {
      available_actions = which(board == 0)
      valid_result = FALSE
      while(!valid_result) {
        if(interactive()) {
          action = readline("Select position (1-9): ")
        } else { # Need this to work in docker
          cat("Select position (1-9): ")
          action = readLines("stdin", n = 1)
        }
        action = suppressWarnings(as.integer(action))
          if(action %in% available_actions) {
            valid_result = TRUE
          } else {
            cat("\nPlease enter a valid action\n")
          }
        }
      return(action)
    }
  )
  
  class(out) = "agent"
  return(out)
}