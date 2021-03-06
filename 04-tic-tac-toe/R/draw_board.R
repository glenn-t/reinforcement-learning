draw_board = function(board, instructions = FALSE) {
  # Set up symbols
  symbols = rep("  ", length(board))
  symbols[board == 1] = "x "
  symbols[board == 2] = "o "
  
  if(instructions) {
    symbols = paste0(c(7:9, 4:6, 1:3), " ")
  }
  
  cat("\n")
  cat("---------\n")
  cat("| ", symbols[1:3], "|\n", sep = "")
  cat("| ", symbols[4:6], "|\n", sep = "")
  cat("| ", symbols[7:9], "|\n", sep = "")
  cat("---------\n\n")
}

draw_instructions = function() {
  draw_board(1:9, instructions = TRUE)
}