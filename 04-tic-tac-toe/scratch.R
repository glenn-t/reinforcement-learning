library(dplyr)
# Source all functions
for(f in list.files("R", full.names = TRUE)) {
  source(f)
}

# Generate all states
all_states = get_all_states_and_winner()

# # Human vs human
# human = new_human()
# draw_board(1:9, instructions = TRUE)
# play_game(list(human, human), draw = TRUE, all_states)
# 

# Human vs agent
# agent_random = new_agent_random()
# human = new_human()
# draw_board(1:9, instructions = TRUE)
# play_game(list(intelligent_agent, human), draw = TRUE, all_states)


#### Train an agent against random agent

# Set up agents
human = new_human()
agent_random = new_agent_random()
agent_p1 = new_agent_01(all_states, symbol = 1)
agent_p2 = new_agent_01(all_states, symbol = 2)

print("Your turn - play 5 games and see if it gets better")
draw_instructions()
out = train_agents(1, human, agent_p2, draw = TRUE)
