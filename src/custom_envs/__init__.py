from gymnasium.envs.registration import register

register(
     id="MovieRec-v1",
     entry_point="src.custom_envs.movie_rec:MovieRecEnv",
)