# train.py
# 학습 스크립트 (하이퍼파라미터, 루프 포함)

import torch
import numpy as np
from envs.vehicle_env import VehicleFollowingEnv
from algo.ppo import PPOAgent
from config import Config # config.py에서 설정 로드
import os
import matplotlib.pyplot as plt # 시각화를 위해 추가

# 롤아웃 데이터를 저장할 클래스
class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.next_states = []

    def add(self, state, action, log_prob, reward, done, next_state):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_states.append(next_state)

    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.next_states = []

if __name__ == "__main__":
    config = Config() # 설정 로드

    # 모델 및 플롯 저장 디렉토리 생성
    os.makedirs(config.model_dir, exist_ok=True)
    os.makedirs(config.plot_dir, exist_ok=True)

    # 여러 환경 생성
    envs = [VehicleFollowingEnv() for _ in range(config.num_envs)]
    
    # PPO 에이전트 생성
    state_dim = envs[0].observation_space.shape[0]
    action_dim = envs[0].action_space.shape[0]
    agent = PPOAgent(state_dim, action_dim, config)

    # 로그 저장용 리스트
    episode_rewards = []
    episode_collisions = []

    # 각 환경의 현재 상태
    current_states = [env.reset()[0] for env in envs] # reset()은 (observation, info) 튜플을 반환
    
    # 학습 루프
    for episode in range(config.num_episodes):
        rollout_buffer = RolloutBuffer()
        total_episode_reward = 0
        episode_collision_count = 0 # 에피소드당 충돌 횟수

        for t in range(config.rollout_len): # 일정 타임스텝마다 PPO 업데이트
            actions = []
            log_probs = []
            
            # 각 환경에서 액션 선택
            for i in range(config.num_envs):
                action, log_prob = agent.select_action(current_states[i])
                actions.append(action)
                log_probs.append(log_prob)
            
            next_states = []
            rewards = []
            dones = []
            
            # 각 환경에서 스텝 진행
            for i, env in enumerate(envs):
                next_state, reward, done, _, _ = env.step(actions[i]) # (obs, reward, terminated, truncated, info)
                next_states.append(next_state)
                rewards.append(reward)
                dones.append(done)

                rollout_buffer.add(current_states[i], actions[i], log_probs[i], reward, done, next_state)

                if done:
                    # 충돌 여부 확인 (distance < 0이면 충돌)
                    if env.distance < 0:
                        episode_collision_count += 1
                    
                    # 에피소드가 끝나면 환경 리셋
                    current_states[i], _ = env.reset() # reset()은 (observation, info) 튜플을 반환
                else:
                    current_states[i] = next_state
                
                total_episode_reward += reward / config.num_envs # 평균 보상 계산

        # PPO 업데이트
        agent.update(rollout_buffer)
        rollout_buffer.clear() # 버퍼 비우기

        episode_rewards.append(total_episode_reward)
        # 모든 환경에서 충돌이 한 번이라도 발생했다면 해당 에피소드는 충돌로 기록 (0 또는 1)
        episode_collisions.append(1 if episode_collision_count > 0 else 0)

        # 로그 출력
        if (episode + 1) % config.log_interval == 0:
            avg_reward = np.mean(episode_rewards[-config.log_interval:])
            avg_collision = np.mean(episode_collisions[-config.log_interval:])
            print(f"Episode: {episode+1}/{config.num_episodes}, Avg Reward ({config.log_interval} eps): {avg_reward:.2f}, Avg Collision ({config.log_interval} eps): {avg_collision:.2f}")

        # 모델 저장
        if (episode + 1) % config.save_interval == 0:
            torch.save(agent.policy_net.state_dict(), os.path.join(config.model_dir, f"policy_net_episode_{episode+1}.pth"))
            torch.save(agent.value_net.state_dict(), os.path.join(config.model_dir, f"value_net_episode_{episode+1}.pth"))
            print(f"모델 저장 완료: episode_{episode+1}.pth")


    print("학습 완료!")

    # 학습 결과 시각화
    # 보상 그래프
    plt.figure(figsize=(12, 6))
    plt.plot(episode_rewards)
    plt.title("Episode Rewards Over Time")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.savefig(os.path.join(config.plot_dir, "episode_rewards.png"))
    plt.show()

    # 충돌 횟수 그래프
    plt.figure(figsize=(12, 6))
    plt.plot(episode_collisions)
    plt.title("Episode Collisions Over Time (1: Collision, 0: No Collision)")
    plt.xlabel("Episode")
    plt.ylabel("Collision Status")
    plt.yticks([0, 1])
    plt.grid(True)
    plt.savefig(os.path.join(config.plot_dir, "episode_collisions.png"))
    plt.show()
