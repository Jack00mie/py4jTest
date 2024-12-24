import os
import signal

from py4j.java_gateway import JavaGateway, CallbackServerParameters
from stable_baselines3 import DQN
import py4JEnvironment
import time
import threading


def exit_app():
    print("Killing process.")
    time.sleep(0.5)
    os.kill(os.getpid(), signal.SIGINT)

class Py4JTest(object):
    gateway: JavaGateway = None

    def test(self, observationVectorSize: int) -> str:
        print("Py4J test round started.")
        env = py4JEnvironment.TestPy4JEnv(observationVectorSize)
        model = DQN("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=1000, log_interval=4)
        print("Py4J Test round complete.")
        threading.Thread(target=exit_app).start()
        return "Py4J Test round complete."

    class Java:
        implements = ["environment.application.py4JApplication.Py4JTest"]

# Make sure that the python code is started first.
# Then execute: java -cp py4j.jar py4j.examples.SingleThreadClientApplication

if __name__ == '__main__':
    py4j_test = Py4JTest()
    gateway = JavaGateway(
        callback_server_parameters=CallbackServerParameters(),
        python_server_entry_point=py4j_test)
    py4j_test.gateway = gateway
    print("Py4J Server started.")
    # while True:
    #    time.sleep(1)


