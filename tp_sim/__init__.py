from gym.envs.registration import register

register(
    id='tp_sim-v0',
    entry_point='tp_sim.envs:TPSimEnv',
)