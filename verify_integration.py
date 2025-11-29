import sys
import os
import numpy as np

# Add root to path
sys.path.append(os.getcwd())

from utils.envs.vehicle_env import VehicleFollowingEnv

def verify():
    print("Initializing Environment...")
    env = VehicleFollowingEnv()
    
    print(f"Observation Space: {env.observation_space}")
    assert env.observation_space.shape == (5,), f"Expected shape (5,), got {env.observation_space.shape}"
    
    print("Resetting Environment...")
    obs, info = env.reset()
    print(f"Initial Observation: {obs}")
    print(f"Info: {info}")
    
    # Check if severity is in observation (index 4)
    severity = obs[4]
    print(f"Initial Severity: {severity}")
    
    print("Stepping Environment...")
    action = [0.5]
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Next Observation: {obs}")
    print(f"Reward: {reward}")
    print(f"Info: {info}")
    
    print("Verification Successful!")

if __name__ == "__main__":
    verify()
