import cProfile, pstats  # import cProfile, pstats
import random

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

if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.strip_dirs()
    stats.print_stats()
    stats.dump_stats('./tests/cProfile/cp')
