import numpy as np
import gymnasium as gym
import requests
from stable_baselines3 import DQN
from fastapi import BackgroundTasks, FastAPI

# start with: uvicorn httpEnvironment:app --host 0.0.0.0 --port 8095

JAVA_PORT = 8094
OBSERVATION_VECTOR_SIZE = 200

class HttpTest():
    def test(self):
        env = TestHttpEnv(OBSERVATION_VECTOR_SIZE)
        model = DQN("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=1000, log_interval=4)
        requests.post(f"http://127.0.0.1:{JAVA_PORT}/testComplete")
        print("test complete")
        exit()

class TestHttpEnv(gym.Env):

    def __init__(self, observation_vector_size: int):
        self.observation_vector_size = observation_vector_size

        self.observation_space = gym.spaces.Box(low=-100_000, high=100_000, dtype=np.float32, shape=(observation_vector_size,))
        self.action_space = gym.spaces.Discrete(4)

    def reset(self, *, seed: int | None = None, options = None,):
        super().reset(seed=seed)

        response = requests.post(f"http://127.0.0.1:{JAVA_PORT}/reset")
        response_body = response.json()

        observation_vector = np.array(response_body["observationVector"])
        info = response_body["info"]
        return observation_vector, info

    def step(self, action: int):
        response = requests.post(f"http://127.0.0.1:{JAVA_PORT}/step", json={"action": int(action)})
        response_body = response.json()

        observation_vector = np.array(response_body["observationVector"])
        reward = response_body["reward"]
        terminated = response_body["terminated"]
        truncated = response_body["truncated"]
        info = response_body["info"]
        return observation_vector, reward, terminated, truncated, info

app = FastAPI()

@app.post("/start")
async def start(background_tasks: BackgroundTasks):
    test = HttpTest()
    background_tasks.add_task(test.test)
    return "Test started."
