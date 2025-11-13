# utils/plot.py
# 학습 과정 및 결과 시각화 함수

import matplotlib.pyplot as plt
import os
import numpy as np

def plot_rewards(rewards, save_path):
    """
    에피소드별 총 보상 그래프를 그립니다.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(rewards)
    plt.title("Episode Rewards Over Time")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.savefig(save_path)
    print(f"보상 그래프 저장 완료: {save_path}")

def plot_collision_rate(collisions, window, save_path):
    """
    최근 N 에피소드 기준 충돌률 moving average를 그립니다.
    """
    if len(collisions) < window:
        print(f"충돌률 이동 평균을 계산하기에 에피소드 수가 부족합니다 (현재: {len(collisions)}, 필요: {window})")
        return

    moving_avg = np.convolve(collisions, np.ones(window)/window, mode='valid')

    plt.figure(figsize=(12, 6))
    plt.plot(moving_avg)
    plt.title(f"Collision Rate Moving Average (Window: {window})")
    plt.xlabel("Episode (after window)")
    plt.ylabel("Collision Rate")
    plt.grid(True)
    plt.ylim(0, 1) # 충돌률은 0에서 1 사이
    plt.savefig(save_path)
    print(f"충돌률 이동 평균 그래프 저장 완료: {save_path}")

# 필요하다면 다른 시각화 함수를 추가할 수 있습니다.