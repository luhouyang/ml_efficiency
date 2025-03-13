# https://python.plainenglish.io/memory-efficient-python-code-slash-memory-usage-by-50-with-this-one-trick-a2124172787b

from memory_profiler import profile
import timeit


class RegularPoint:

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class SlottedPoint:
    __slots__ = ['x', 'y', 'z']

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


from dataclasses import dataclass

@dataclass
class DataClsPoint:
    __slots__ = ['x', 'y', 'z']
    x: float
    y: float
    z: float


def access_regular():
    p = RegularPoint(1, 2, 3)
    x = p.x
    y = p.y
    z = p.z


def access_slotted():
    p = SlottedPoint(1, 2, 3)
    x = p.x
    y = p.y
    z = p.z


def access_datacls():
    p = DataClsPoint(1, 2, 3)
    x = p.x
    y = p.y
    z = p.z


def timetest():
    print("Regular class:", timeit.timeit(access_regular, number=1_000_000))

    print("Slotted class:", timeit.timeit(access_slotted, number=1_000_000))

    print("Data class:", timeit.timeit(access_datacls, number=1_000_000))


@profile
def basecls():
    arr = []
    for _ in range(1_000_000):
        arr.append(RegularPoint(1, 2, 3))


@profile
def slottedcls():
    arr = []
    for _ in range(1_000_000):
        arr.append(SlottedPoint(1, 2, 3))


@profile
def datacls():
    arr = []
    for _ in range(1_000_000):
        arr.append(DataClsPoint(1, 2, 3))
