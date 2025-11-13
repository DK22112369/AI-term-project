# AI TermProject — Vehicle Following (PPO)

This repository contains a lightweight skeleton for training a PPO agent on a
1D vehicle-following environment. Isaac Sim integration points are deliberately
left as TODO placeholders so you can plug in real simulation calls later.

Quick start

1. Create a Python environment and install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2. Run a smoke test to verify basic imports and env step:

```powershell
python smoke_test.py
```

Notes
- `utils/envs/vehicle_env.py` implements a Gym-style environment using simple
  1D kinematics. All Isaac Sim API calls are marked with `TODO:` comments.
- `algo/ppo.py` contains a minimal PPO skeleton. `PPOAgent.update()` is a
  placeholder and should be implemented for full training behavior.
- `train.py` contains a multi-environment training loop that uses `PPOAgent`.
