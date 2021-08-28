import pathlib
import shutil
import time

import ray
import tensorflow as tf
import numpy as np
import click

from buffer import ReplayBuffer
from workers import Actor, Learner


@click.command()
@click.option('--env_name', type=str, default="CartPole-v1")
@click.option('--num_actors', type=int, default=4)
@click.option('--num_iters', type=int, default=30000)
@click.option("--logdir", type=click.Path(file_okay=False), default="log")
@click.option("--cluster", is_flag=True)
def main(env_name, num_actors, num_iters, logdir, cluster):

    logdir = pathlib.Path(logdir)
    if logdir.exists():
        shutil.rmtree(logdir)

    summary_writer = tf.summary.create_file_writer(str(logdir))

    if not cluster:
        ray.init()

    epsilons = np.linspace(0.01, 0.8, num_actors)

    print("==== ACTORS launch ====")
    actors = [Actor.remote(pid=i, env_name=env_name, epsilon=epsilons[i])
              for i in range(num_actors)]

    replaybuffer = ReplayBuffer(buffer_size=2**15)

    print("==== LEARNER launch ====")
    learner = Learner.remote(env_name=env_name)

    current_weights = ray.put(ray.get(learner.get_weights.remote()))

    print("==== TESTER launch ====")
    tester = Actor.remote(pid=None, env_name=env_name, epsilon=0.0)

    wip_actors = [actor.rollout.remote(current_weights) for actor in actors]

    n = 0

    print("==== Initialize buffer ====")
    for _ in range(50):
        finished_actor, wip_actors = ray.wait(wip_actors, num_returns=1)
        td_errors, transitions, pid = ray.get(finished_actor[0])
        replaybuffer.add(td_errors, transitions)
        wip_actors.extend([actors[pid].rollout.remote(current_weights)])
        n += 1

    minibatchs = [replaybuffer.sample_minibatch(batch_size=512) for _ in range(16)]

    wip_learner = learner.update_network.remote(minibatchs)

    minibatchs = [replaybuffer.sample_minibatch(batch_size=512) for _ in range(16)]

    wip_tester = tester.test_play.remote(current_weights)

    t = time.time()
    lap_count = 0

    while n <= num_iters:

        finished_actor, wip_actors = ray.wait(wip_actors, num_returns=1, timeout=0)

        if finished_actor:
            td_errors, transitions, pid = ray.get(finished_actor[0])
            replaybuffer.add(td_errors, transitions)
            wip_actors.extend([actors[pid].rollout.remote(current_weights)])
            n += 1
            lap_count += 1

        finished_learner, _ = ray.wait([wip_learner], num_returns=1, timeout=0)

        if finished_learner:

            current_weights, indices, td_errors, loss_info = ray.get(finished_learner[0])

            wip_learner = learner.update_network.remote(minibatchs)

            current_weights = ray.put(current_weights)

            replaybuffer.update_priority(indices, td_errors)

            minibatchs = [replaybuffer.sample_minibatch(batch_size=512) for _ in range(16)]

            with summary_writer.as_default():
                tf.summary.scalar("Buffer", len(replaybuffer), step=n)
                tf.summary.scalar("loss", loss_info, step=n)
                tf.summary.scalar("lap_count", lap_count, step=n)
                tf.summary.scalar("lap_time", time.time() - t, step=n)

            t = time.time()
            lap_count = 0

        if n % 200 == 0:
            test_score = ray.get(wip_tester)
            wip_tester = tester.test_play.remote(current_weights)
            with summary_writer.as_default():
                tf.summary.scalar("test_score", test_score, step=n)


if __name__ == '__main__':
    main()
