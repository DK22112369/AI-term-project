import torch


class Config:
	def __init__(self):
		# 환경 관련 하이퍼파라미터
		self.dt = 0.1  # 시간 간격 (s)
		self.v_lead_initial = 20.0  # 선행차 초기 속도 (m/s)
		self.v_lead_brake_decel = -5.0 # 선행차 감속 시 감가속도 (m/s^2)
		self.v_target = 15.0 # 목표 속도 (m/s)
		self.safety_distance = 10.0  # 안전 거리 (m)
		self.T_max = 200  # 에피소드 최대 길이 (step 수)
		self.initial_distance = 30.0 # 초기 차간 거리 (m)
		self.ego_initial_velocity = 15.0 # 내 차 초기 속도 (m/s)

		# 보상 함수 가중치
		self.reward_alpha = 0.1  # 효율 보상 가중치
		self.reward_beta = 0.05  # 승차감 보상 가중치

		# PPO 에이전트 관련 하이퍼파라미터
		self.gamma = 0.99
		self.gae_lambda = 0.95 # GAE 람다 값
		self.clip_param = 0.2
		self.policy_lr = 3e-4
		self.value_lr = 1e-3
		self.ppo_epochs = 10
		self.batch_size = 64
		# Loss coefficients
		self.entropy_coef = 0.01
		self.value_loss_coef = 0.5
		# Network sizes
		self.policy_hidden_sizes = (64, 64)
		self.value_hidden_sizes = (64, 64)
		# Initial policy std (log_std init)
		self.policy_log_std_init = -0.5

		# 학습 루프 관련 하이퍼파라미터
		self.num_envs = 4  # 여러 환경을 동시에 돌릴 개수
		self.num_episodes = 1000 # 총 에피소드 수
		self.rollout_len = 200 # PPO 업데이트 주기 (몇 스텝마다 업데이트할지)
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		# 로그 및 저장 관련
		self.log_interval = 10 # 몇 에피소드마다 로그를 출력할지
		self.save_interval = 100 # 몇 에피소드마다 모델을 저장할지
		self.model_dir = "./models" # 모델 저장 경로
		self.plot_dir = "./plots" # 그래프 저장 경로

