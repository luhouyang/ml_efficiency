from emnist import EMNIST
from memory_profiler import profile
import line_profiler


@profile
def load_emnist():
    ds = EMNIST(root=r'C:\Users\User\Desktop\Python\ml_efficiency\archive')


load_emnist()

timeprofile = line_profiler.LineProfiler()


@timeprofile
def time_load_emnist():
    ds = EMNIST(root=r'C:\Users\User\Desktop\Python\ml_efficiency\archive')


time_load_emnist()
timeprofile.print_stats()
