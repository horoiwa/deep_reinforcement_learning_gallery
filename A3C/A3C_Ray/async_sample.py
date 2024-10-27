import time
import random

import ray


@ray.remote
class Agent:

    def __init__(self, agent_id):

        self.agent_id = agent_id

    def run_task(self):
        n = random.randint(2, 5)
        time.sleep(n)
        return (n, self.agent_id)


def main(num_updates=10):

    ray.init()
    agents = [Agent.remote(agent_id=i) for i in range(4)]

    total = 0
    work_in_progresses = [agent.run_task.remote() for agent in agents]
    for _ in range(num_updates):
        finished_job, work_in_progresses = ray.wait(work_in_progresses, num_returns=1)
        res, agent_id = ray.get(finished_job)[0]
        total += res
        work_in_progresses.extend([agents[agent_id].run_task.remote()])

    ray.shutdown()
    print(total)


if __name__ == "__main__":
    main()
