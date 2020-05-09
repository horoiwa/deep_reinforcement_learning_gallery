import subprocess


def gridsearch():
    import itertools

    lrs = [0.001, 0.0005, 0.0001]
    copy_periods = [1000, 500, 200]

    for copy_period, lr in itertools.product(copy_periods, lrs):
        print("START", lr, copy_period)
        subprocess.call(["python", "main.py", str(lr), str(copy_period)])


if __name__ == "__main__":
    gridsearch()
