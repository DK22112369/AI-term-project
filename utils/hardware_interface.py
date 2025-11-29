import time
import numpy as np

class CarInterface:
    """
    Hardware Interface for Jetson Nano based Mini-Car.
    Handles communication with sensors (Camera, LiDAR) and actuators (Motor, Servo).
    """
    def __init__(self, is_simulation=False):
        self.is_simulation = is_simulation
        
        if not self.is_simulation:
            # TODO: Initialize actual hardware libraries here
            # e.g., from jetracer.nvidia_racecar import NvidiaRacecar
            # self.car = NvidiaRacecar()
            print("[INFO] Initializing Hardware Interface...")
        else:
            print("[INFO] Initializing Mock Interface for Simulation...")

        # State variables
        self.distance = 30.0
        self.velocity = 0.0
        self.last_time = time.time()

    def get_sensor_data(self):
        """
        Retrieve data from sensors.
        Returns:
            dict: {
                "distance": float (meters),
                "velocity": float (m/s),
                "image": np.array (if camera used)
            }
        """
        if self.is_simulation:
            # Mock data for testing
            return self._get_mock_data()
        
        # TODO: Implement actual sensor reading
        # 1. Read Camera -> Object Detection -> Distance Estimation
        # 2. Read LiDAR (if available) -> Distance
        # 3. Read IMU/Encoder -> Velocity
        
        return {
            "distance": self.distance, # Placeholder
            "velocity": self.velocity,
            "image": None
        }

    def apply_control(self, action):
        """
        Apply control actions to the car.
        Args:
            action (float): Acceleration/Brake command [-1.0, 1.0]
        """
        throttle = float(np.clip(action, -1.0, 1.0))
        
        if not self.is_simulation:
            # TODO: Map action to PWM signals
            # self.car.throttle = throttle
            # self.car.steering = 0.0 (Keep straight for now)
            pass
        else:
            # Update mock state
            dt = time.time() - self.last_time
            self.last_time = time.time()
            self.velocity += throttle * 3.0 * dt # Simple physics
            self.velocity = max(0.0, self.velocity)
            print(f"[Mock Control] Action: {throttle:.2f}, Velocity: {self.velocity:.2f}")

    def _get_mock_data(self):
        # Simulate some sensor noise
        return {
            "distance": 30.0 + np.random.normal(0, 0.1),
            "velocity": self.velocity,
            "image": None
        }

    def close(self):
        if not self.is_simulation:
            # Stop motors
            pass
