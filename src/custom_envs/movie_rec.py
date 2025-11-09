import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class MovieRecEnv(gym.Env):
    """
    Session-based movie recommendation environment where the goal is to maximize
    cumulative reward from a user session.
    A user's preference is simulated based on pre-defined personas.
    """
    def __init__(self):
        super(MovieRecEnv, self).__init__()

        # --- 1. Define Movie Catalog & Genres (Inspired by MovieLens) ---
        # For simplicity, we create a mock catalog of 20 movies with genres.
        # In a real scenario, this would be loaded from movies.csv.
        self.movie_catalog = {
            0: {"name": "Action Movie 1", "genres": ["Action", "Adventure"]},
            1: {"name": "Action Movie 2", "genres": ["Action", "Thriller"]},
            2: {"name": "Sci-Fi Movie 1", "genres": ["Sci-Fi", "Adventure"]},
            3: {"name": "Sci-Fi Movie 2", "genres": ["Sci-Fi", "IMAX"]},
            4: {"name": "Comedy Movie 1", "genres": ["Comedy"]},
            5: {"name": "Comedy Movie 2", "genres": ["Comedy", "Romance"]},
            6: {"name": "Romance Movie 1", "genres": ["Romance", "Drama"]},
            7: {"name": "Romance Movie 2", "genres": ["Romance"]},
            8: {"name": "Horror Movie 1", "genres": ["Horror", "Thriller"]},
            9: {"name": "Horror Movie 2", "genres": ["Horror", "Mystery"]},
            10: {"name": "Drama Movie 1", "genres": ["Drama"]},
            11: {"name": "Documentary 1", "genres": ["Documentary"]},
            12: {"name": "Kids Movie 1", "genres": ["Children", "Animation"]},
            13: {"name": "Action Comedy", "genres": ["Action", "Comedy"]},
            14: {"name": "Sci-Fi Action", "genres": ["Sci-Fi", "Action"]},
            15: {"name": "Romantic Comedy", "genres": ["Romance", "Comedy"]},
            16: {"name": "Action Drama", "genres": ["Action", "Drama"]},
            17: {"name": "Sci-Fi Drama", "genres": ["Sci-Fi", "Drama"]},
            18: {"name": "Historical Drama", "genres": ["Drama", "War"]},
            19: {"name": "Fantasy Adventure", "genres": ["Fantasy", "Adventure"]},
        }
        self.genres = sorted(list(set(genre for movie in self.movie_catalog.values() for genre in movie["genres"])))
        self.genre_map = {genre: i for i, genre in enumerate(self.genres)}
        self.num_genres = len(self.genres)
        self.num_movies = len(self.movie_catalog)

        # --- 2. Define User Personas (Inspired by MovieLens ratings) ---
        self.user_personas = {
            "action_fan": {"fav_genres": ["Action", "Adventure", "Sci-Fi"], "hate_genres": ["Romance", "Documentary"]},
            "romance_lover": {"fav_genres": ["Romance", "Comedy", "Drama"], "hate_genres": ["Horror", "Action", "War"]},
            "sci-fi_geek": {"fav_genres": ["Sci-Fi", "IMAX", "Mystery"], "hate_genres": ["Comedy", "Children"]},
        }

        # --- 3. Define Action and Observation Space (Standard Gym Interface) ---
        self.action_space = spaces.Discrete(self.num_movies)
        # Observation: [satisfaction_per_genre..., fatigue_level, session_step]
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_genres + 2,), dtype=np.float32)
        
        self.max_session_length = 20 # A session ends after 20 recommendations

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # --- Start a new session with a new user ---
        # 1. Select a random user persona
        persona_name, self.current_user_persona = random.choice(list(self.user_personas.items()))
        
        # 2. Reset user state
        self.satisfaction_history = np.zeros(self.num_genres, dtype=np.float32)
        self.fatigue = 0.0
        self.session_step = 0
        
        observation = self._get_observation()
        info = {"user_persona": persona_name}
        
        return observation, info

    def step(self, action):
        assert self.action_space.contains(action), "Invalid Action"

        self.session_step += 1
        
        # --- Simulate user's response to the recommended movie (action) ---
        recommended_movie = self.movie_catalog[action]
        
        # 1. Calculate click probability based on persona
        click_prob = self._calculate_click_prob(recommended_movie)
        
        # 2. Determine if the user clicks
        reward = 0
        is_clicked = random.random() < click_prob

        if is_clicked:
            reward = 1.0
            # Update satisfaction history for the genres of the clicked movie
            for genre in recommended_movie["genres"]:
                self.satisfaction_history[self.genre_map[genre]] += 0.2
            # Good recommendations reduce fatigue
            self.fatigue = max(0.0, self.fatigue - 0.1)
        else:
            reward = -0.2 # Penalty for a bad recommendation
            # Bad recommendations increase fatigue
            self.fatigue += 0.2

        # 3. Determine if the user churns (leaves the session)
        terminated = False
        if self.fatigue >= 1.0:
            terminated = True
            reward = -10.0 # Large penalty for losing the user

        # 4. Check if session length is exceeded
        truncated = False
        if self.session_step >= self.max_session_length:
            truncated = True

        done = terminated or truncated
        observation = self._get_observation()
        
        # Find the persona name for logging
        persona_name = "unknown"
        for name, persona_details in self.user_personas.items():
            if persona_details == self.current_user_persona:
                persona_name = name
                break

        info = {
            "is_clicked": is_clicked, 
            "fatigue": self.fatigue, 
            "click_prob": click_prob,
            "user_persona": persona_name,
            "recommended_movie": recommended_movie['name'],
            "churned": terminated and self.fatigue >= 1.0
        }

        return observation, reward, done, False, info # Return done for both terminated and truncated

    def _get_observation(self):
        # Normalize satisfaction and fatigue to be between 0 and 1
        norm_satisfaction = np.clip(self.satisfaction_history, 0, 1)
        norm_session_step = self.session_step / self.max_session_length
        
        # Concatenate to form the final observation vector
        obs = np.concatenate([norm_satisfaction, [self.fatigue], [norm_session_step]]).astype(np.float32)
        return obs

    def _calculate_click_prob(self, movie):
        base_prob = 0.3 # Base probability of clicking anything
        
        # Increase probability for favorite genres
        for genre in movie["genres"]:
            if genre in self.current_user_persona["fav_genres"]:
                base_prob += 0.3
        
        # Decrease probability for hated genres
        for genre in movie["genres"]:
            if genre in self.current_user_persona["hate_genres"]:
                base_prob -= 0.2
        
        # Fatigue also reduces click probability
        prob = base_prob - self.fatigue * 0.5
        
        return max(0.05, min(prob, 0.95)) # Clip probability to a realistic range

    def render(self, mode='human'):
        pass # Not needed for this environment
