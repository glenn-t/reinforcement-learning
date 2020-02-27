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
agent_p1 = new_agent_01(all_states, symbol = 1, eps = 100, alpha = 0.99)
agent_p2 = new_agent_01(all_states, symbol = 2, eps = 1, alpha = 0.5)

test_agents = function(N, p1, p2) {
  winner = replicate(N, {
    play_game(list(p1, p2), draw = FALSE, all_states)$winner
  })
  return(table(winner))
}

# Train p1
out = train_agents(1500, agent_p1, agent_random)
agent_p1 = out$p1

# Train p2
out = train_agents(1500, agent_random, agent_p2)
agent_p2 = out$p2

# Train against each other
out = train_agents(1000, agent_p1, agent_p2)
agent_p1 = out$p1
agent_p2 = out$p2

# Get average win rate against other agent
cat("\nagent_p1 against random\n")
print(test_agents(1000, agent_p1, agent_random))

cat("\nrandom against agent_p2\n")
print(test_agents(1000, agent_random, agent_p2))

cat("\nagent_p1 against agent_p2\n")
print(test_agents(1000, agent_p1, agent_p2))

# # Learn against each other
# out = train_agents(1, agent_p1, agent_p2)
# agent_p1 = out$p1
# agent_p2 = out$p2

# And again
cat("\nagent_p1 against random\n")
print(test_agents(1000, agent_p1, agent_random))

cat("\nrandom against agent_p2\n")
print(test_agents(1000, agent_random, agent_p2))

cat("\nagent_p1 against agent_p2\n")
print(test_agents(1000, agent_p1, agent_p2))


# Now play against human or watch a game
print("Agent vs agent game")
play_game(list(agent_p1, agent_p2), draw = TRUE, all_states)

# print("Your turn")
# draw_instructions()
# play_game(list(human, agent_p2), draw = TRUE, all_states)
# 
# print("Your turn")
# draw_instructions()
# play_game(list(agent_p1, human), draw = TRUE, all_states)

# Train it yourself!
print("Your turn - play 5 games and see if it gets better")
draw_instructions()
out = train_agents(1, human, agent_p2, draw = TRUE)
# agent_p1 = out$p1
agent_p2 = out$p2

print("Training fresh agents against each other")
N = 5000
agent_p1 = new_agent_01(all_states, symbol = 1, alpha = 0.5, eps = 100)
agent_p2 = new_agent_01(all_states, symbol = 2, alpha = 0.5, eps = 100)
out = train_agents(N, agent_p1, agent_p2)
agent_p1 = out$p1
agent_p2 = out$p2

out = train_agents(N, agent_random, agent_p2)
agent_p2 = out$p2

out = train_agents(N, agent_p1, agent_random)
agent_p1 = out$p1

# And again
cat("\nagent_p1 against random\n")
print(test_agents(1000, agent_p1, agent_random))

cat("\nrandom against random\n")
print(test_agents(1000, agent_random, agent_p2))

cat("\nagent_p1 against agent_p2\n")
print(test_agents(1000, agent_p1, agent_p2))

# Train it yourself!
print("Your turn - play 5 games and see if it gets better")
draw_instructions()
out = train_agents(5, human, agent_p2, draw = TRUE)
# agent_p1 = out$p1
agent_p2 = out$p2




# Human trained
# Agent1 learns well with alpha of 0.99 or 1, but agent2 doesn't.
agent_p1h = new_agent_01(all_states, symbol = 1, alpha = 0.95, eps = 0, initial = 0.5)
out = train_agents(10, agent_p1h, human, draw = TRUE)
agent_p1h = out$p1

# Inspect value function
agent_p1h$value_function %>% 
  left_join(all_states) %>% 
  filter(value != 0.5 & !is_ended) %>% 
  mutate(first_game = (x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9) == 1) %>%
  filter(first_game) %>%
  arrange(value) 
