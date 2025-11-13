"""Simple smoke test: import env and run one reset/step."""
import numpy as np

from utils.envs.vehicle_env import VehicleFollowingEnv


def main():
    env = VehicleFollowingEnv()
    s, info = env.reset()
    print("reset ok", s)
    a = np.array([0.0], dtype=np.float32)
    ns, r, done, truncated, info = env.step(a)
    print("step ok", ns, r, done, truncated)


if __name__ == "__main__":
    main()
