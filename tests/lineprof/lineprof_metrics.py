# https://github.com/pyutils/line_profiler

import line_profiler
import random

profile = line_profiler.LineProfiler()

@profile
def main():
    arr = []
    for i in range(4000):
        x = pow(i, 2)
        y = x / (i - 1 + 1e-8)
        arr.append(x)
        arr.append(y)

        generate()


def generate():
    data = [random.randint(0, 99) for p in range(0, 1000)]
    return data

main()
profile.print_stats()