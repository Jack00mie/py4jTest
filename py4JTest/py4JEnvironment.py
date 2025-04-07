import numpy as np
import gymnasium as gym
import py4j.java_gateway
from py4j.java_gateway import JavaGateway


class TestPy4JEnv(gym.Env):

    def __init__(self, observation_vector_size: int):
        self.observation_vector_size = observation_vector_size

        self.gateway = JavaGateway()
        self.java_environment = self.gateway.entry_point.getEnvironment(self.observation_vector_size)

        self.observation_space = gym.spaces.MultiDiscrete(np.full(observation_vector_size, 100))
        self.action_space = gym.spaces.Discrete(4)

    def reset(self, *, seed: int | None = None, options = None,):
        super().reset(seed=seed)
        first_observation = self.java_environment.reset()
        observation_vector = np.array(first_observation.getObservationVector())
        info = dict(first_observation.getInfo())

        return observation_vector, info

    def step(self, action: int):
        transition = self.java_environment.step(int(action))

        return np.array(transition.getObservationVector()), transition.getReward(), transition.getTerminated(), transition.getTruncated(), dict(transition.getInfo())

    def close(self):
        py4j.java_gateway.quiet_shutdown(self.java_environment)

