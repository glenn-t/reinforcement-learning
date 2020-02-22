library(dplyr)
# Source all functions
lapply(list.files("R", full.names = TRUE), source)

# Generate all states
all_states = get_all_states_and_winner()

# Human vs human
human = new_human()
draw_board(1:9, instructions = TRUE)
play_game(list(human, human), draw = TRUE, all_states)

# Random vs Random
agent_random = new_agent_random()
draw_board(1:9, instructions = TRUE)
play_game(list(agent_random, agent_random), draw = TRUE, all_states)

