from gymnasium.envs.registration import register

register(
    id='Grid-v0',
    entry_point='src.env.grid:GridEnv',
    max_episode_steps=300,
)