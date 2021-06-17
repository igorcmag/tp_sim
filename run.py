import gym
import tp_sim
#from agents.agent import amostrar
import agents.agent

# Simulation parameters
# Total number of players
N = 5
# Space dimension
D = 5
# Number of timesteps
T = 10
# Vector of naive agents bids 
mu_b = [i+1 for i in range(N-1)]
# Vector of naive agents asks
mu_a = [i+1 for i in range(N-1)]
# Standard deviation in naive agents noise bids and asks
sigma = 0
# Government price
gov_price = 2

env = gym.make('tp_sim-v0', N = N, D = D, T = T, mu_b = mu_b, mu_a = mu_a, sigma = sigma, gov_price = gov_price)
agent = agents.agent.TPSimAgent(D = D, T = T, mission = env.missions[0])
print(agent.mission)

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
print(" ############### BEGINNING ##############")
print('action_book')
print(env.action_book)
print('book')
print(env.book)
print('in_poss')
print(env.in_poss)
print('owners')
print(env.owners)
print('credits')
print(env.credits)
print('last transactions')
print(env.last_trans)

action = agent.take_action(state)
print(" ############### AGENT INFO ##############")
print('book_prices')
print(agent.book_prices)
print('in_poss')
print(agent.in_poss)
print('credits')
print(agent.credit)
print('last transactions')
print(agent.last_trans)


print(" ############### FIRST STEP ##############")
action = agent.take_action(state)

print(" ############### ENV INFO ##############")
state, reward, done, info = env.step(action)
print('action_book')
print(env.action_book)
print('book')
print(env.book)
print('in_poss')
print(env.in_poss)
print('owners')
print(env.owners)
print('credits')
print(env.credits)
print('last transactions')
print(env.last_trans)

action = agent.take_action(state)

print(" ############### AGENT INFO ##############")
print('book_prices')
print(agent.book_prices)
print('in_poss')
print(agent.in_poss)
print('credits')
print(agent.credit)
print('last transactions')
print(agent.last_trans)

#print(info)
#action = agents.agent.take_action(state)
#print(env.action_book)
#print(env.book)
#print(env.missions)
#n_state, reward, done, info = env.step(action)
#print(env.action_book)