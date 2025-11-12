import matplotlib
matplotlib.use('Agg') #플롯을 화면에 표시하는 대신 파일로 저장하기 위해 필요.
import matplotlib.pyplot as plt
import torch
import os
import numpy as np

def plot_training_progress(episode_durations, episode_rewards, results_dir, episode_index, quiet=False):
    """
    ### 학습 과정 시각화
    
    #### 설계 의도 (Design Intent)
    - 학습 상태 모니터링: 에피소드별 보상과 길이를 시각적으로 표현하여, 에이전트의 학습이
      올바른 방향으로 진행되고 있는지(e.g., 보상이 증가하는 추세인지)를 직관적으로 파악.
    - 추세 분석: 단일 에피소드의 값은 변동성이 크므로, 이동 평균(Moving Average)을 함께 표시하여
      전반적인 성능 향상 추세를 더 명확하게 분석.
      
    Args:
        episode_durations (list): 각 에피소드의 길이를 담은 리스트.
        episode_rewards (list): 각 에피소드의 총 보상을 담은 리스트.
        results_dir (str): 플롯 이미지를 저장할 디렉토리 경로.
        episode_index (int): 현재 에피소드 번호 (파일 이름에 사용).
        quiet (bool): True이면 플롯을 화면에 표시하지 않음.
    """
    # 플롯을 그릴 Figure 객체 생성. 2개의 서브플롯(1행 2열)을 가짐.
    plt.figure(figsize=(12, 6))

    # --- 1. 에피소드 길이(Duration) 플롯 ---
    plt.subplot(1, 2, 1)
    plt.title(f'Episode Durations (Episode {episode_index})')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(episode_durations, label='Duration')

    # 데이터가 100개 이상 쌓이면 100-에피소드 이동 평균을 계산하여 함께 표시.
    if len(episode_durations) >= 100:
        # `unfold`를 사용하여 100개씩 묶인 텐서를 만들고, 각 묶음의 평균을 계산.
        moving_average_durations = torch.tensor(episode_durations, dtype=torch.float32).unfold(0, 100, 1).mean(1).view(-1)
        # 이동 평균은 100번째 에피소드부터 시작하므로, 앞 99개는 0으로 채워 길이를 맞춤.
        moving_average_durations = torch.cat((torch.zeros(99), moving_average_durations))
        plt.plot(moving_average_durations.numpy(), label='Moving Average (100)')
    plt.legend()

    # --- 2. 에피소드 보상(Reward) 플롯 ---
    plt.subplot(1, 2, 2)
    plt.title(f'Episode Rewards (Episode {episode_index})')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(episode_rewards, label='Reward', color='green')

    if len(episode_rewards) >= 100:
        moving_average_rewards = torch.tensor(episode_rewards, dtype=torch.float32).unfold(0, 100, 1).mean(1).view(-1)
        moving_average_rewards = torch.cat((torch.zeros(99), moving_average_rewards))
        plt.plot(moving_average_rewards.numpy(), label='Moving Average (100)', color='red')
    plt.legend()
    
    # 플롯 레이아웃을 자동으로 조정하여 제목이나 라벨이 겹치지 않도록 함.
    plt.tight_layout()
    
    # --- 3. 플롯 저장 및 표시 ---
    # 지정된 경로에 플롯 이미지를 저장.
    plot_path = os.path.join(results_dir, f'training_progress_episode_{episode_index}.png')
    plt.savefig(plot_path)
    
    # `quiet` 모드가 아닐 경우, 플롯을 화면에 표시.
    if not quiet:
        plt.show()
        
    # 메모리 누수를 방지하기 위해 현재 플롯을 닫음.
    plt.close()
