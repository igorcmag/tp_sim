N = 10
D = 10

def amostrar(env):
    return env.action_space.sample()

def take_action(state):
    return [[10 for i in range(D**2)]]