from pylinal import Matrix
from tadgrad.conv import *


def test_im_to_channels():
    im = Matrix([
        [(1, 5, 9), (2, 6, 10)],
        [(3, 7, 11), (4, 8, 12)]
    ])
    m1 = [
        [1, 2],
        [3, 4]
    ]
    m2 = [
        [5, 6],
        [7, 8]
    ]
    m3 = [
        [9, 10],
        [11, 12]
    ]
    channels = list(im_to_channels(im))
    expect = [Matrix(m1), Matrix(m2), Matrix(m3)]
    assert channels == expect, channels
    print("im_to_channels: success")
    return


def test_im_from_channels():
    m1 = [
        [1, 2],
        [3, 4]
    ]
    m2 = [
        [5, 6],
        [7, 8]
    ]
    m3 = [
        [9, 10],
        [11, 12]
    ]
    maps = (m for m in [m1, m2, m3])
    im = im_from_channels(maps)
    assert im == Matrix([
        [(1, 5, 9), (2, 6, 10)],
        [(3, 7, 11), (4, 8, 12)]
    ])
    print("im_from_channels: success")
    return


def test_conv2():
    m = Matrix([
        [7, 6, 5, 5, 6, 7],
        [6, 4, 3, 3, 4, 6],
        [5, 3, 2, 2, 3, 5],
        [5, 3, 2, 2, 3, 5],
        [6, 4, 3, 3, 4, 6],
        [7, 6, 5, 5, 6, 7]
    ])
    kernel = Matrix([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    conv = conv2(kernel)
    result = conv(m)

    expect = Matrix([
        [9, 8, 6, 6, 8, 9],
        [8, 2, 1, 1, 2, 8],
        [6, 1, 0, 0, 1, 6],
        [6, 1, 0, 0, 1, 6],
        [8, 2, 1, 1, 2, 8],
        [9, 8, 6, 6, 8, 9],
    ])
    assert result == expect
    print("conv2: success")
    return 


def test_conv3():
    image = Matrix([
        [(1, 2, 3), (4, 5, 6), (7, 8, 9)],
        [(10, 11, 12), (13, 14, 15), (16, 17, 18)]
    ])
    m1 = [
        [1, 4, 7],
        [10, 13, 16]
    ]
    m2 = [
        [2, 5, 8],
        [11, 14, 17]
    ]
    m3 = [
        [3, 6, 9],
        [12, 15, 18]
    ]

    kernel = Matrix([
        [1, 2],
        [3, 4]
    ])
    conv = conv2(kernel)
    
    maps = [conv(m1), conv(m2), conv(m3)]
    expect = im_from_channels(maps)

    conv = conv3(kernel)
    result = conv3(kernel)(image)
    
    assert result == expect
    print("conv3: success")
    return


def test():
    test_conv2()
    test_im_from_channels()
    test_im_to_channels()
    test_conv3()
    return


if __name__ == '__main__':
    test()
