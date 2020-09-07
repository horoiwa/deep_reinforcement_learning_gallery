import ray


@ray.remote
class Counter(object):
    def __init__(self):
        self.x = 0

    def inc(self):
        self.x += 1

    def get_value(self):
        return self.x


def main():
    ray.init()
    # Create an actor process.
    c = Counter.remote()

    # Check the actor's counter value.
    print(ray.get(c.get_value.remote()))  # 0

    # Increment the counter twice and check the value again.
    c.inc.remote()
    c.inc.remote()
    print(ray.get(c.get_value.remote()))  # 2

    ray.shutdown()


if __name__ == "__main__":
    main()
