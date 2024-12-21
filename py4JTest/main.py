from py4j.java_gateway import JavaGateway, CallbackServerParameters
from stable_baselines3 import DQN
import py4JEnvironment


class Py4JTest(object):
    def test(self, observationVectorSize: int) -> str:
        env = py4JEnvironment.TestPy4JEnv(observationVectorSize)
        model = DQN("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=1000, log_interval=4)
        return "test complete"

    class Java:
        implements = ["environment.application.py4JApplication.Py4JTest"]

# Make sure that the python code is started first.
# Then execute: java -cp py4j.jar py4j.examples.SingleThreadClientApplication

if __name__ == '__main__':
    py4j_test = Py4JTest()
    gateway = JavaGateway(
        callback_server_parameters=CallbackServerParameters(),
        python_server_entry_point=py4j_test)
    print("Server started.")


