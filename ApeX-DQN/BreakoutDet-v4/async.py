import time
import threading
import random
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import time
import random

import util


def bullshit_job(i):
    time.sleep(random.randint(0, 3))
    return i


def main():
    with util.Timer("no async"):
        jobs = [bullshit_job(i) for i in range(10)]
        print(jobs)

    with util.Timer("tpe"):
        tpe = ThreadPoolExecutor(max_workers=3)

if __name__ == "__main__":
    main()
