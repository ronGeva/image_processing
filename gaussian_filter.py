import sys
import os
import cv2
import matplotlib.pyplot as plt
import numpy
import math


DEFAULT_SIGMA = 1
DEFAULT_KERNEL_SIZE = 3


def gaussian_filter(x: int, y: int, s: float):
    """
    Calculates the gaussian filter for parameters x, y and s (sigma).
    """
    return numpy.exp(-(x ** 2 + y ** 2) / (2 * (s ** 2))) / (2 * numpy.pi * (s ** 2))


def get_gaussian_matrix(size: int, sigma: float):
    """
    Generates and returns a (size x size) matrix containing the values of a gaussian filter.
    """
    assert size % 2 == 1, 'Filter matrix\'s size should be odd'
    matrix_center = (size // 2, size // 2)
    mat = numpy.ndarray(shape=(size, size))
    for row_index in range(size):
        for col_index in range(size):
            mat[row_index][col_index] = gaussian_filter(row_index - matrix_center[0], col_index - matrix_center[1],
                                                        sigma)
    return mat


def smooth_using_gaussian_fft(img: numpy.ndarray, sigma: float, kernel_size: int):
    """
    Cause a "blurring" effect in the input image by applying a gaussian filter with the parameters supplied.
    :return: The output image (after the filter).
    """
    gaussian_mat = get_gaussian_matrix(kernel_size, sigma)
    frequency_gaussian_mat = numpy.fft.fft2(gaussian_mat, s=img.shape)
    frequency_img = numpy.fft.fft2(img)
    frequency_blurred_img = frequency_gaussian_mat * frequency_img
    time_blurred_img = numpy.fft.ifft2(frequency_blurred_img)
    for row_index in range(time_blurred_img.shape[0]):
        for col_index in range(time_blurred_img.shape[1]):
            time_blurred_img[row_index][col_index] = round(math.fabs(time_blurred_img[row_index][col_index]))
    return time_blurred_img.real


def parse_input(args):
    assert 2 <= len(args) <= 4, "Usage: python ex1.py <input file> [<sigma>] [<size of filter's kernel>] (optionals)"
    input_file = args[1]
    assert os.path.isfile(input_file), "First argument ({0}) isn't a valid path!".format(input_file)
    if len(args) >= 3:
        sigma = args[2]
        try:
            sigma = float(sigma)  # no built-in way to test for float compatibility in strings
        except ValueError:
            assert False, "Second argument ({0}) isn't a number!".format(sigma)
        sigma = float(sigma)
    else:
        sigma = DEFAULT_SIGMA

    if len(args) == 4:
        kernel_size = args[3]
        assert kernel_size.isnumeric(), "Third argument ({0}) isn't a number!".format(kernel_size)
        kernel_size = int(kernel_size)
    else:
        kernel_size = DEFAULT_KERNEL_SIZE
    return input_file, sigma, kernel_size


def main(args):
    try:
        input_file, sigma, filter_kernel_size = parse_input(args)
    except AssertionError as e:
        print(e.args[0])
        return
    img = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
    plt.imshow(img, cmap='gray')
    plt.show()
    img = smooth_using_gaussian_fft(img, sigma, filter_kernel_size)
    plt.imshow(img, cmap='gray')
    plt.show()


main(sys.argv)
