from gym.envs.registration import register

register(
    id='text_on_image-v0',
    entry_point='delta_gym.envs:TextOnImageEnv',
    max_episode_steps=14,
)
