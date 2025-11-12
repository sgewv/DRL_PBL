import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class MovieRecEnv(gym.Env):
    """
    ### 세션 기반 영화 추천을 위한 커스텀 강화학습 환경
    
    #### 설계 의도 (Design Intent)
    - 문제 정의: 사용자가 서비스를 이탈(churn)하지 않고, 한 세션(session) 내에서 최대한 많은 영화를 보도록
      연속적으로 영화를 추천하는 순차적 추천(sequential recommendation) 문제를 시뮬레이션.
    - 목표: 사용자의 만족도를 최대화하고 피로도를 최소화하여 세션의 길이를 늘리고, 궁극적으로 누적 보상(cumulative reward)을 최대화하는 추천 정책을 학습.
    
    #### MDP(Markov Decision Process) 정의
    - 상태(State): 현재 사용자의 상태를 나타내는 벡터.
      `[장르별 만족도 벡터, 현재 피로도, 현재 세션 진행률]`
      - 장르별 만족도: 사용자가 특정 장르의 영화를 클릭할 때마다 해당 장르의 만족도가 증가. 사용자의 장기적인 선호를 모델링.
      - 피로도(Fatigue): 사용자가 관심 없는 영화를 추천받을수록 증가. 피로도가 임계치를 넘으면 사용자는 세션을 이탈(churn).
      - 세션 진행률: 현재 세션이 얼마나 진행되었는지를 나타냄.
    - 행동(Action): 영화 카탈로그에 있는 `num_movies`개의 영화 중 하나를 사용자에게 추천.
    - 보상(Reward):
      - 클릭 시: `+1.0` (긍정적 피드백)
      - 클릭하지 않을 시: `-0.2` (약한 부정적 피드백)
      - 이탈(Churn) 시: `-10.0` (강한 부정적 피드백)
    - 에피소드 종료(Done):
      - Terminated: 사용자의 피로도가 1.0 이상이 되어 세션을 이탈(churn)하는 경우.
      - Truncated: 세션이 최대 길이(`max_session_length`)에 도달한 경우.
    """
    def __init__(self):
        super(MovieRecEnv, self).__init__()

        # === 하위 목표 1: 환경의 기본 요소 정의 ===
        
        # --- 1. 영화 카탈로그 및 장르 정의 ---
        # 실제 시나리오에서는 MovieLens 데이터셋의 `movies.csv` 파일 등에서 로드.
        # 여기서는 간단한 시뮬레이션을 위해 20개의 영화와 장르를 하드코딩.
        self.movie_catalog = {
            0: {"name": "액션 영화 1", "genres": ["Action", "Adventure"]},
            1: {"name": "액션 영화 2", "genres": ["Action", "Thriller"]},
            2: {"name": "SF 영화 1", "genres": ["Sci-Fi", "Adventure"]},
            3: {"name": "SF 영화 2", "genres": ["Sci-Fi", "IMAX"]},
            4: {"name": "코미디 영화 1", "genres": ["Comedy"]},
            5: {"name": "코미디 영화 2", "genres": ["Comedy", "Romance"]},
            6: {"name": "로맨스 영화 1", "genres": ["Romance", "Drama"]},
            7: {"name": "로맨스 영화 2", "genres": ["Romance"]},
            8: {"name": "공포 영화 1", "genres": ["Horror", "Thriller"]},
            9: {"name": "공포 영화 2", "genres": ["Horror", "Mystery"]},
            10: {"name": "드라마 영화 1", "genres": ["Drama"]},
            11: {"name": "다큐멘터리 1", "genres": ["Documentary"]},
            12: {"name": "아동 영화 1", "genres": ["Children", "Animation"]},
            13: {"name": "액션 코미디", "genres": ["Action", "Comedy"]},
            14: {"name": "SF 액션", "genres": ["Sci-Fi", "Action"]},
            15: {"name": "로맨틱 코미디", "genres": ["Romance", "Comedy"]},
            16: {"name": "액션 드라마", "genres": ["Action", "Drama"]},
            17: {"name": "SF 드라마", "genres": ["Sci-Fi", "Drama"]},
            18: {"name": "역사 드라마", "genres": ["Drama", "War"]},
            19: {"name": "판타지 어드벤처", "genres": ["Fantasy", "Adventure"]},
        }
        self.genres = sorted(list(set(genre for movie in self.movie_catalog.values() for genre in movie["genres"])))
        self.genre_map = {genre: i for i, genre in enumerate(self.genres)}
        self.num_genres = len(self.genres)
        self.num_movies = len(self.movie_catalog)

        # --- 2. 사용자 페르소나 정의 ---
        # 사용자의 다양한 취향을 시뮬레이션하기 위한 가상의 사용자 그룹(페르소나).
        # 각 페르소나는 선호하는 장르와 싫어하는 장르를 가짐.
        self.user_personas = {
            "action_fan": {"fav_genres": ["Action", "Adventure", "Sci-Fi"], "hate_genres": ["Romance", "Documentary"]},
            "romance_lover": {"fav_genres": ["Romance", "Comedy", "Drama"], "hate_genres": ["Horror", "Action", "War"]},
            "sci-fi_geek": {"fav_genres": ["Sci-Fi", "IMAX", "Mystery"], "hate_genres": ["Comedy", "Children"]},
        }

        # --- 3. 행동 및 관찰 공간 정의 (Gymnasium 표준 인터페이스) ---
        # 행동 공간(Action Space): 에이전트가 할 수 있는 행동의 집합. 이산적인 영화 ID.
        self.action_space = spaces.Discrete(self.num_movies)
        # 관찰 공간(Observation Space): 에이전트가 환경으로부터 받는 상태 정보의 형태와 범위.
        # [장르별 만족도 벡터, 피로도, 세션 진행률]로 구성된 연속적인 벡터.
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_genres + 2,), dtype=np.float32)
        
        self.max_session_length = 20 # 한 세션의 최대 추천 횟수

    def reset(self, seed=None, options=None):
        """
        ### 설계 의도 (Design Intent)
        - 새로운 에피소드(세션)를 시작하기 위해 환경을 초기 상태로 리셋.
        - 매번 다른 사용자와의 상호작용을 시뮬레이션하기 위해, 사용자 페르소나를 무작위로 선택.
        """
        super().reset(seed=seed)
        
        # === 하위 목표: 새로운 사용자 세션 시작 ===
        # 1. 사용자 페르소나를 무작위로 선택하여 이번 세션의 사용자 취향 결정.
        persona_name, self.current_user_persona = self.np_random.choice(list(self.user_personas.items()))
        
        # 2. 사용자의 상태(만족도, 피로도 등)를 초기화.
        self.satisfaction_history = np.zeros(self.num_genres, dtype=np.float32)
        self.fatigue = 0.0
        self.session_step = 0
        
        observation = self._get_observation()
        info = {"user_persona": persona_name} # 디버깅 및 분석을 위한 추가 정보
        
        return observation, info

    def step(self, action):
        """
        ### 설계 의도 (Design Intent)
        - 에이전트가 선택한 행동(영화 추천)에 대한 환경(사용자)의 반응을 시뮬레이션.
        - 추천 결과에 따라 상태를 업데이트하고, 보상을 계산하여 에이전트에게 피드백을 제공.
        """
        assert self.action_space.contains(action), "유효하지 않은 행동입니다."

        self.session_step += 1
        
        # === 하위 목표: 추천 행동에 대한 사용자 반응 시뮬레이션 ===
        recommended_movie = self.movie_catalog[action]
        
        # --- 1. 페르소나 기반 클릭 확률 계산 ---
        click_prob = self._calculate_click_prob(recommended_movie)
        
        # --- 2. 클릭 여부 결정 및 보상/상태 업데이트 ---
        reward = 0
        is_clicked = self.np_random.random() < click_prob

        if is_clicked:
            # 긍정적 피드백: 높은 보상, 피로도 감소, 관련 장르 만족도 증가
            reward = 1.0
            for genre in recommended_movie["genres"]:
                self.satisfaction_history[self.genre_map[genre]] += 0.2
            self.fatigue = max(0.0, self.fatigue - 0.1)
        else:
            # 부정적 피드백: 작은 페널티, 피로도 증가
            reward = -0.2
            self.fatigue += 0.2

        # --- 3. 사용자 이탈(Churn) 여부 결정 ---
        terminated = False
        if self.fatigue >= 1.0:
            # 피로도가 임계치를 넘으면 사용자는 세션을 종료. 큰 페널티 부여.
            terminated = True
            reward = -10.0

        # --- 4. 최대 세션 길이 도달 여부 확인 ---
        truncated = False
        if self.session_step >= self.max_session_length:
            truncated = True

        done = terminated or truncated
        observation = self._get_observation()
        
        persona_name = next((name for name, p in self.user_personas.items() if p == self.current_user_persona), "unknown")

        info = {
            "is_clicked": is_clicked, 
            "fatigue": self.fatigue, 
            "click_prob": click_prob,
            "user_persona": persona_name,
            "recommended_movie": recommended_movie['name'],
            "churned": terminated and self.fatigue >= 1.0
        }

        return observation, reward, done, False, info

    def _get_observation(self):
        """현재 사용자 상태를 정규화하여 에이전트가 인식할 상태 벡터(관찰)로 변환."""
        norm_satisfaction = np.clip(self.satisfaction_history, 0, 1)
        norm_session_step = self.session_step / self.max_session_length
        
        obs = np.concatenate([norm_satisfaction, [self.fatigue], [norm_session_step]]).astype(np.float32)
        return obs

    def _calculate_click_prob(self, movie):
        """
        사용자 페르소나와 현재 피로도를 바탕으로, 추천된 영화를 클릭할 확률을 계산.
        - 선호 장르일수록 확률 증가.
        - 비선호 장르일수록 확률 감소.
        - 피로도가 높을수록 확률 감소.
        """
        base_prob = 0.3 # 기본 클릭 확률
        
        for genre in movie["genres"]:
            if genre in self.current_user_persona["fav_genres"]:
                base_prob += 0.3
        
        for genre in movie["genres"]:
            if genre in self.current_user_persona["hate_genres"]:
                base_prob -= 0.2
        
        prob = base_prob - self.fatigue * 0.5
        
        return max(0.05, min(prob, 0.95)) # 확률을 [0.05, 0.95] 범위로 클리핑

    def render(self, mode='human'):
        # 이 환경은 시각적 렌더링이 필요 없음.
        pass
