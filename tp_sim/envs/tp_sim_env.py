import gym
from collections import OrderedDict
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete
import pandas as pd
import numpy as np
import random
import tp_sim.envs.naive

# Auxiliary functions
# Convert permit number to permit tuple
def number_to_tuple(D, n):
    return (D % n, D // n)

# Convert permit tuple to permit number
def tuple_to_number(D, permit):
    return permit[1]*D + permit[0]

# Do transactions
def do_transactions(N, D, book, action_book, in_poss, owners):
    for i in range(len(book)):
        if not np.isnan(book.seller[i]):

            # Bids

            # If permit owner made a bid, than discart it
            old_owner = owners[i]
            
            action_book.


            best_bid = np.max(action_book.iloc[i][1:])

            # Only for bids bigger than minimum price
            if best_bid >= book.price[i]:
                # if player is agent, than N --> 0
                player_best_bid = np.argmax(action_book.iloc[i][1:])%N

                # Update possessions
                in_poss[player_best_bid].append(number_to_tuple(i))
                old_owner = owners[i]
                # If old_owner is not government, than remove this possession
                if old_owner != N:
                    in_poss[old_owner].remove(number_to_tuple(i))
                
                # Update owners
                owners[i] = player_best_bid
                book.seller[i] = np.nan
            
            # Asks

    pass

# Evaluate reward
def reward_eval(state):
    return 0

# Reset book
def reset_book(permits, gov_price):
    book_df = pd.DataFrame(columns = ['permit', 'seller', 'price'])
    book_df['permit'] = permits
    book_df['seller'] = 'gov'
    book_df['price'] = gov_price
    return book_df

# Reset action book
def reset_action_book(N, permits):
    action_book_columns = [f'n{x}' for x in range(0, N+1)]
    action_book_columns[0], action_book_columns[-1] = 'permit', 'agent'
    action_book_df = pd.DataFrame(columns = action_book_columns)
    action_book_df['permit'] = permits
    return action_book_df

# Reset missions
def reset_missions(N, permits):
    initial_points = random.sample(permits, N)
    final_points = random.choices(permits, k = N)
    missions = [[initial_points[i], final_points[i]] for i in range(N)]
    return missions

# Reset state state
def reset_state(N, D, mission, gov_price):
    state = {
        'mission': (tuple_to_number(D, mission[0]), tuple_to_number(D, mission[1])), 
        'book': {
            'seller': [N + 1 for i in range(D**2)], 
            'price': [[gov_price for i in range(D**2)]]
            }, 
        'in_poss': [0 for i in range(D**2)], 
        'timestamp': 0
        }
    return state

class TPSimEnv(gym.Env):
    metadata = {'render.modes': ['human']} #?

    def __init__(self, N, D, T, mu_b, mu_a, sigma, gov_price):
        # Set parameters
        # Set total number of players
        self.N = N
        # Set space dimensions
        self.D = D
        # Set vector of naive agents bids
        self.mu_b = mu_b
        # Set vector of naive agents asks
        self.mu_a = mu_a
        # Set standard deviation in naive agents noise bids and asks
        self.sigma = sigma
        # Set number max of iterations per episode
        self.T = T
        # Set timestamp
        self.t = 0
        # Set governemnt price
        self.gov_price = gov_price

        # Action is a vector of D^2 real numbers between low and high
        self.action_space = Box(low = -1000, high = 1000, shape = (1, self.D**2))
        # State is a dictionary of other space tipes 
        self.observation_space = Dict({
            'mission': Tuple((Discrete(self.D**2), Discrete(self.D**2))), 
            'book': Dict({
                'seller': MultiDiscrete([self.N + 2 for i in range(self.D**2)]), 
                'price': Box(low = 0, high = 1000, shape = (1, self.D**2))}), 
            'in_poss': MultiBinary(self.D**2), 
            'timestamp': Discrete(self.T)})

        # Set auxiliary variables
        # Permits vector
        self.permits = []
        for i in range(0, D):
            for j in range(0, D):
                self.permits.append((i, j))

        # Set in_poss
        self.in_poss = [[] for i in range(self.N)]
        # Set owners
        self.owners = [self.N for i in range(self.D**2)]
        # Sort random missions for all players
        self.missions = reset_missions(self.N, self.permits)
        # Start book
        self.book = reset_book(self.permits, self.gov_price)
        # Start action book (empty)
        self.action_book = reset_action_book(self.N, self.permits)
        self.state = reset_state(self.N, self.D, self.missions[0], self.gov_price)


    def step(self, action):
        # Update action_book witg naive actions 
        for i in range(1, self.N):
            self.action_book[f'n{i}'] = tp_sim.envs.naive.take_action_naive(self.missions[i], self.book['permit'], self.in_poss[i], self.mu_b[i-1], self.mu_a[i-1], self.D)[0]
        
        # Update action_book with agent action
        self.action_book[f'agent'] = action[0]
        
        # Do transactions and evaluate next state 
        self.in_poss, self.book = do_transactions(self.N, self.D, self.book, self.action_book, self.in_poss)
        
        # Increase timestamp by 1
        self.t += 1

        # Calculate reward
        reward = reward_eval(self.state)

        # Check if episode is done
        if self.t <= 0:
            done = True
        else:
            done = False

        # Set place holder for info
        info = {'action_book': self.action_book, 'book': self.book}

        # Return step information
        return self.state, reward, done, info

    def render(self, mode='human', close=False):
        # Implement visulization
        pass
    
    def reset(self):
        # Reset all players missions
        #
        # Reset step time
        self.t = self.T
        # Reset in_poss
        self.in_poss = [[] for i in range(self.N)]
        # Reset owners
        self.owners = [self.N for i in range(self.D**2)]
        # Sort random missions for all players
        self.missions = reset_missions(self.N, self.permits)
        # Reset book
        self.book = reset_book(self.permits, self.gov_price)
        # Reset action book
        self.action_book = reset_action_book(self.N, self.permits)
        self.state = reset_state(self.N, self.D, self.missions[0], self.gov_price)

        return self.state
    