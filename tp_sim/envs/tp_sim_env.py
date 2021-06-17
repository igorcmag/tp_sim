import gym
from collections import OrderedDict
from gym.spaces import Discrete, Box, Dict, MultiBinary
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
def do_transactions(N, D, book, action_book, in_poss, owners, credits):
    last_trans = [0 for i in range(D**2)]
    
    for i in range(len(book)):
        # Bids
        if not np.isnan(book.seller[i]):

            bids = action_book.iloc[i][1:]

            # If permit owner is not gov and it made a bid, than discart it
            old_owner = owners[i]
            
            if old_owner != N and bids[old_owner] > 0:
                bids[old_owner] = -1

            best_bid = np.max(bids)

            # Only for bids bigger than minimum price
            if best_bid >= book.price[i]:
                # Find who did best bid
                
                players_best_bid = np.argwhere((bids == best_bid).tolist()).flatten().tolist()

                # Choince someone randomly (in case of a draw)
                player_best_bid = random.choice(players_best_bid)

                # Update possessions
                in_poss[player_best_bid].append(i)
                old_owner = owners[i]
                # If old_owner is not government, than remove this possession
                if old_owner != N:
                    in_poss[old_owner].remove(i)
                
                # Update owner
                owners[i] = player_best_bid

                # Update book
                book.loc[i, 'seller'] = np.nan
                book.loc[i, 'price'] = np.nan
                # Update credits
                # If old_owner is not gov, than give it credits
                if old_owner != N:
                    credits[old_owner] += best_bid
                credits[player_best_bid] -= best_bid

                # Update last transactions
                last_trans[i] = best_bid

        # Ask
        # If permit owner is not gov
        owner = owners[i]
        if owner != N:
            # If permit owner wants to sell it
            asks = action_book.iloc[i][1:]
            if asks[owner] < 0:
                book.loc[i, 'price'] = np.abs(asks[owner])
                book.loc[i, 'seller'] = owner

    return last_trans

# Evaluate reward
def reward_eval(state):
    return 0

# Reset book
def reset_book(N, permits, in_poss, gov_price):
    book_df = pd.DataFrame(columns = ['permit', 'seller', 'price'])
    book_df['permit'] = permits
    book_df['seller'] = N
    book_df['price'] = gov_price

    # Remove players initial points from book
    for i in range(len(in_poss)):
        book_df.loc[in_poss[i][0], 'seller'] = np.nan
        book_df.loc[in_poss[i][0], 'price'] = np.nan 
    return book_df

# Reset action book
def reset_action_book(N, permits):
    action_book_columns = ['permit', 'agent'] + [f'n{x}' for x in range(1, N)]
    action_book_df = pd.DataFrame(columns = action_book_columns)
    action_book_df['permit'] = permits
    return action_book_df

# Reset missions
def reset_missions(N, permits):
    initial_points = random.sample(permits, N)
    final_points = random.choices(permits, k = N)
    missions = [[initial_points[i], final_points[i]] for i in range(N)]
    return missions

# Reset in_poss
def reset_in_poss_owners(N, D, missions):
    in_poss = []
    owners = [N for i in range(D**2)]
    for i in range(len(missions)):
        in_poss.append([tuple_to_number(D, missions[i][0])])
        owners[tuple_to_number(D, missions[i][0])] = i
    return in_poss, owners

# Reset state state
def update_state(book, last_trans, in_poss, credit, t):
    state = { 
        'book_prices': [book.price.tolist()], 
        'in_poss': [in_poss], 
        'credit': [credit],
        'last_trans': [last_trans],
        'timestamp': t
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
            'book_prices': Box(low = 0, high = 1000, shape = (1, self.D**2)), 
            'in_poss': MultiBinary(self.D**2),
            'credit': Box(low = -100000, high = 100000, shape = (1,1)),
            'last_trans':  Box(low = 0, high = 1000, shape = (1, self.D**2)),
            'timestamp': Discrete(self.T)
            })

        # Set auxiliary variables
        # Permits vector
        self.permits = []
        for i in range(0, D):
            for j in range(0, D):
                self.permits.append((i, j))

        # Set credits
        self.credits = [0 for i in range(self.N)]
        # Sort random missions for all players
        self.missions = reset_missions(self.N, self.permits)
        # Set in_poss and owners
        self.in_poss, self.owners = reset_in_poss_owners(self.N, self.D, self.missions)
        # Start book
        self.book = reset_book(self.N, self.permits, self.in_poss, self.gov_price)
        # Start last transactions
        self.last_trans = [0 for i in range(self.D**2)]
        # Start action book (empty)
        self.action_book = reset_action_book(self.N, self.permits)
        self.state = update_state(self.book, self.last_trans, self.in_poss[0], self.credits[0], 0)


    def step(self, action):
        # Update action_book witg naive actions 
        for i in range(1, self.N):
            #self.action_book[f'n{i}'] = tp_sim.envs.naive.take_action_naive(self.missions[i], self.book['permit'], self.in_poss[i], self.mu_b[i-1], self.mu_a[i-1], self.D)
            self.action_book[f'n{i}'] = tp_sim.envs.naive.Naive.random_action(self.D, -10, 10)
        
        # Update action_book with agent action
        self.action_book[f'agent'] = action[0]
        
        # Do transactions 
        self.last_trans = do_transactions(self.N, self.D, self.book, self.action_book, self.in_poss, self.owners, self.credits)
        
        # Update state
        self.state = update_state(self.book, self.last_trans, self.in_poss[0], self.credits[0], self.t + 1)

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
        # Set credits
        self.credits = [0 for i in range(self.N)]
        # Sort random missions for all players
        self.missions = reset_missions(self.N, self.permits)
        # Reset in_poss and owners
        self.in_poss, self.owners = reset_in_poss_owners(self.N, self.D, self.missions)
        # Reset book
        self.book = reset_book(self.N, self.permits, self.in_poss, self.gov_price)
        # Reset last transactions
        self.last_trans = [0 for i in range(self.D**2)]
        # Reset action book
        self.action_book = reset_action_book(self.N, self.permits)
        self.state = update_state(self.book, self.last_trans, self.in_poss[0], self.credits[0], 0)

        return self.state
    