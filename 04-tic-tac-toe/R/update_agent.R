update_agent = function(agent, state_history) {
  UseMethod("update_agent")
}

# Method to produce an updated agent for updatable_agent class
update_agent.updatable_agent = function(agent, state_history) {
  return(agent$update_function(agent, state_history))
}

# If agent is not updatable, then just return the agent
update_agent.default = function(agent, state_history) {
  return(agent)
}