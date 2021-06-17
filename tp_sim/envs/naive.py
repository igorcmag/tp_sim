from random import randint

# mission: player's mission
# av_for_purch: all permits that are available for purchase at book
# in_poss: all permits naive player has now
# buy_price: how much naive player pay for a permit
# sell_price: by how much naive player sell a permit
# D: space dimension


class Naive():
    def __init__(self):
        pass

    def random_action(D, min, max):
        action = []
        for i in range(D**2):
            action.append(randint(min, max))

        return action
    
    def take_action_naive(mission, av_for_purch, in_poss, buy_price, sell_price, D):
        pass
