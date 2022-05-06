from typing import Callable, Union, Iterator, Iterable, List
from pylinal import Matrix


T = Union[float, int, complex]
Reduction = Callable[[Matrix], T]
Conv = Callable[[Matrix], Matrix]


def pointwise_mult(lhs: Matrix, rhs: Matrix) -> Matrix:
    result: Matrix = Matrix(
        (x*y for x, y in zip(lv, rv))
        for lv, rv in zip(lhs, rhs)
    )
    return result


def sum_reduction(m: Matrix) -> T:
    return sum(sum(row) for row in m)


def padded(m: Matrix) -> Matrix:
    copy = [[el for el in row] for row in m]
    length = len(copy)

    up = [copy[0][0]] + copy[0] + [copy[0][-1]]
    bottom = [copy[-1][0]] + copy[-1] + [copy[-1][-1]]
    copy = [up] + copy + [bottom]

    for i in range(1, length + 1):
        copy[i] = [copy[i][0]] + copy[i] + [copy[i][-1]]

    return Matrix(copy)


def conv2(
    kernel: Matrix[T],
    padding: bool = True,
    reduction: Reduction = sum_reduction,
) -> Conv:
    rows, cols = kernel.shape

    def _reduce(minor: Matrix) -> T:
        return reduction(pointwise_mult(minor, kernel))

    def closure(m: Matrix) -> Matrix:
        if padding:
            m = padded(m)

        horizontal_steps = 1 + m.shape[1] - cols
        vertical_steps = 1 + m.shape[0] - rows

        elements = (
            (_reduce(m[r:r+rows, c:c+cols]) for c in range(horizontal_steps))  # type: ignore
            for r in range(vertical_steps)
        )

        return Matrix(elements)

    return closure


def im_from_channels(channels: Iterable[Matrix]) -> Matrix:
    zip_ = zip(*channels)
    it: Iterator[Iterator[tuple]] = (
        zip(*rows)
        for rows in zip_
    )
    return Matrix(it)


def im_to_channels(m: Matrix[tuple]) -> Iterator[Matrix]:  # type: ignore
    in_channels: int = len(m[-1][-1])  # type: ignore
    channels: Iterator[Matrix] = (
        Matrix(
            (el[c] for el in row)
            for row in m
        )
        for c in range(in_channels)
    )
    return channels


def conv3(
    kernel: Matrix[T],
    padding: bool = True,
    reduction: Reduction = sum_reduction,
) -> Conv:
    conv = conv2(
        kernel,
        padding=padding,
        reduction=reduction,
    )

    def closure(m: Matrix[tuple]) -> Matrix[tuple]:  # type: ignore
        channels: Iterator[Matrix] = im_to_channels(m)
        maps: Iterator[Matrix] = (conv(channel) for channel in channels)
        return im_from_channels(maps)

    return closure
