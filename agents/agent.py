from random import randint

# Auxiliary functions
# Convert permit number to permit tuple
def number_to_tuple(D, n):
    return (D % n, D // n)

# Convert permit tuple to permit number
def tuple_to_number(D, permit):
    return permit[1]*D + permit[0]

class TPSimAgent():
    def __init__(self, D, T, mission):
        # Fixed during episode
        self.D = D
        self.T = T
        self.mission = mission

        # Variable during episode
        self.book_prices = None
        self.in_poss = None
        self.credit = None
        self.last_trans = None
        self.timestamp = 0
    
    def update_state(self, state):
        self.book_prices = state['book_prices'][0]
        self.in_poss = state['in_poss'][0]
        self.credit = state['credit'][0]
        self.last_trans = state['last_trans'][0]
        self.timestamp = state['timestamp']

    def take_action(self, state):
        # Update observated state
        self.update_state(state)

        # RL Strategy

        # Take action

        return [[randint(-10, 10) for i in range(self.D**2)]]