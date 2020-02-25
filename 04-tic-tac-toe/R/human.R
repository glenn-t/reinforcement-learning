# Creates a human agent
new_human = function() {
  
  out = list(
    "choose_action" = function(self, board) {
      
      # keyboard mapping
      mapping = c(
        "7" = 1,
        "8" = 2,
        "9" = 3,
        "4" = 4,
        "5" = 5,
        "6" = 6,
        "1" = 7,
        "2" = 8,
        "3" = 9
      )
      
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
        # Convert to board format
        action = as.vector(mapping[as.character(action)])
        
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