library(dplyr)
# Source all functions
for(f in list.files("R", full.names = TRUE)) {
  source(f)
}

# Generate all states
all_states = get_all_states_and_winner()

#### Train an agent against random agent

# Set up agents
human = new_human()
agent_random = new_agent_random()

# Fun things to try
# Agents learn faster and explore more with optimistic initial values (e.g 0.9)
# Interesting to try to make them lose, or prefer to lose than draw

## Things to note
# When training against random agents, the agent may play more agressively in
# order to get more wins and avoid draws, at the cost of a giving the opponent
# a few wins. This tends to happen with low alphas (e.g 0.3).
# Higher alphas tend to produce one's avoid losing.

# Training against random agents is not really a good idea as general strategy
# since they will not play like an adverserial opponent 
# (e.g. won't go for the win). It works ok for tic-tac-toe since they will play
# like a good opponent a small proportion of the time.

agent_p1 = new_agent_01(all_states, symbol = 1, eps = 0, alpha = 0.3, initial = 0.5)
agent_p2 = new_agent_01(all_states, symbol = 2, eps = 0, alpha = 0.3, initial = 0.5)

# Train against each other - # 7000 is maximum larning for this alpha=0.5
# out = train_agents(10, agent_p1, agent_p2, draw = TRUE)
# agent_p1 = out$p1
# agent_p2 = out$p2

out = train_agents(10000, agent_p1, agent_p2)
agent_p1 = out$p1
agent_p2 = out$p2

# Get average win rate against other agent
cat("\nagent_p1 against random\n")
print(test_agents(1000, agent_p1, agent_random))

cat("\nrandom against agent_p2\n")
print(test_agents(1000, agent_random, agent_p2))

cat("\nagent_p1 against agent_p2\n")
print(test_agents(1000, agent_p1, agent_p2))

# Now play against human or watch a game
print("Agent vs agent game")
out = play_game(list(agent_p1, agent_p2), draw = TRUE, all_states)

# Train it yourself!
print("Your turn - play 5 games and see if it gets better")
# draw_instructions()
out = train_agents(5, human, agent_p2, draw = TRUE); agent_p2 = out$p2
print("Switch sides - play as circle")
out = train_agents(5, agent_p1, human, draw = TRUE); agent_p1 = out$p1

# Inspect value function of first turn
print("Value function for first turn")
agent_p2$value_function %>% 
  left_join(all_states) %>% 
  filter(value != 0.5 & !is_ended) %>% 
  mutate(first_game = (x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9) == 3) %>%
  filter(first_game & x5) %>%
  arrange(value) %>%
  print()
