import gym
import tp_sim
#from agents.agent import amostrar
import agents.agent

# Simulation parameters
# Total number of players
N = 10
# Space dimension
D = 10
# Number of timesteps
T = 10
# Vector of naive agents bids 
mu_b = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# Vector of naive agents asks
mu_a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# Standard deviation in naive agents noise bids and asks
sigma = 0
# Government price
gov_price = 2

env = gym.make('tp_sim-v0', N = N, D = D, T = T, mu_b = mu_b, mu_a = mu_a, sigma = sigma, gov_price = gov_price)

# Run experiment
#episodes = T
#for episode in range(1, episodes + 1):
#    state = env.reset()
#    done = False
#    score = 0

#    while not done:
#        #env.render()
#        action = amostrar(env)
#        n_state, reward, done, info = env.step(action)
#        score += reward

#    print(f'Episode: {episode} Score: {score}')


state = env.reset()
print(env.action_book)
print(env.book)
action = agents.agent.take_action(env)
state, reward, done, info = env.step(action)
print(env.action_book)
print(env.book)
#print(info)
#action = agents.agent.take_action(state)
#print(env.action_book)
#print(env.book)
#print(env.missions)
#n_state, reward, done, info = env.step(action)
#print(env.action_book)