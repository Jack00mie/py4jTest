import os
import signal
import time

import numpy as np
import gymnasium as gym
import requests
from pydantic import BaseModel, Field
from stable_baselines3 import DQN
from fastapi import BackgroundTasks, FastAPI

# start with: uvicorn httpEnvironment:app --host 0.0.0.0 --port 8095
# or from root directory: uvicorn httpTest.httpEnvironment:app --host 0.0.0.0 --port 8095

JAVA_PORT = 8094
PORT = 8095


class HttpTest:
    def __init__(self, observation_vector_size: int):
        self.observation_vector_size = observation_vector_size

    def test(self):
        print("Http test round started.")
        env = TestHttpEnv(self.observation_vector_size)
        model = DQN("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=1000, log_interval=4)
        requests.post(f"http://127.0.0.1:{JAVA_PORT}/testComplete", "Http test round complete.")
        print("Http test round complete.")
        exit_app()


class TestHttpEnv(gym.Env):

    def __init__(self, observation_vector_size: int):
        self.observation_vector_size = observation_vector_size

        self.observation_space = gym.spaces.MultiDiscrete(np.full(observation_vector_size, 100))
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


class ObservationVectorSize(BaseModel):
    observationVectorSize: int = Field(gt=0)

@app.post("/start")
async def start(observation_vector_size_class: ObservationVectorSize, background_tasks: BackgroundTasks):
    test = HttpTest(observation_vector_size_class.observationVectorSize)
    background_tasks.add_task(test.test)
    return "Http test round started."


def exit_app():
    print("Killing process.")
    time.sleep(0.5)
    os.kill(os.getpid(), signal.SIGINT)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=PORT)
    print("Http Server started.")
