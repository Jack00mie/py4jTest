import numpy as np
import gymnasium as gym
import requests

JAVA_PORT = 8094

class TestHttpEnv(gym.Env):

    def __init__(self, observation_vector_size: int):
        self.observation_vector_size = observation_vector_size

        self.observation_space = gym.spaces.Box(low=-100_000, high=100_000, dtype=np.float32, shape=(observation_vector_size,))
        self.action_space = gym.spaces.Discrete(4)

    def reset(self, *, seed: int | None = None, options = None,):
        super().reset(seed=seed)

        r = requests.post(f"http://127.0.0.1:{JAVA_PORT}/step", json={"action": 42})
        b = requests.get(f"http://127.0.0.1:{JAVA_PORT}/step")
        print(b)
        print(r.json())
        print(r.headers)
        observation_vector = 1
        return observation_vector, {"info": "info text"}

    def step(self, action: int):
        transition = self.java_environment.step(int(action))

        return np.array(transition.observationVector), transition.reward, transition.terminated, transition.truncated, {"info": transition.info}


t = TestHttpEnv(100)
t.reset()

