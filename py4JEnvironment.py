import numpy as np
import gymnasium as gym
from py4j.java_gateway import JavaGateway, GatewayParameters


class TestPy4JEnv(gym.Env):

    def __init__(self, observation_vector_size: int):
        self.observation_vector_size = observation_vector_size
        self.gateway = JavaGateway(gateway_parameters=GatewayParameters(auto_field=True))
        self.java_environment = self.gateway.entry_point.getEnvironment(self.observation_vector_size)

        self.observation_space = gym.spaces.Box(low=-100_000, high=100_000, dtype=np.float32, shape=(observation_vector_size,))
        self.action_space = gym.spaces.Discrete(4)

    def reset(self, *, seed: int | None = None, options = None,):
        super().reset(seed=seed)
        observation_vector = np.array(self.java_environment.reset())

        return observation_vector, {"info": "info text"}

    def step(self, action: int):
        transition = self.java_environment.step(int(action))

        return np.array(transition.observationVector), transition.reward, transition.terminated, transition.truncated, {"info": transition.info}

