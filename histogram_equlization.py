import cv2
import matplotlib.pyplot as plt
import numpy
from os.path import dirname, abspath, join


def normalize_indexes(shape: (int, int), row: int, col: int):
    """
    Normalizes the row and column indexes to achieve the effect of "mirroring" values around image borders.
    """
    row = int(numpy.fabs(row))
    col = int(numpy.fabs(col))
    if row >= shape[0]:
        row = 2 * shape[0] - 1 - row
    if col >= shape[1]:
        col = 2 * shape[1] - 1 - col
    return row, col


def get_histogram_equlization_map(histogram: list[int], amount_of_pixels: int):
    """
    Given an histogram of the pixels in a specific neighborhood, creates and returns an histogram equalization mapping,
    allowing us to easily map an existing pixel value to a new value which will result in maximal histogram
    equalization.
    :param histogram: A mapping of each pixel value to the amount of pixels with this value.
    :param amount_of_pixels: The total amount of pixels in the histogram.
    :return: A mapping of each previous pixel value to its new value.
    """
    pdf = {}  # probability distribution function
    for key, value in enumerate(histogram):
        pdf[key] = value / amount_of_pixels

    cdf = {}  # cumulative distribution function
    cumulative_distrubtion = 0
    for key, value in pdf.items():
        cumulative_distrubtion += pdf[key]
        cdf[key] = cumulative_distrubtion
    he = {}  # histogram equalization
    for key, value in cdf.items():
        he[key] = round(value * 255)
    return he


def get_local_histogram(img: numpy.ndarray, row: int, col: int, neighborhood_size: int):
    """
    Retrieves a list containing 256 entries, each mapping a pixel value to the amount of pixels with this value in a
    specific neighborhood.
    :param img: The image on which to perform the histogram.
    :param row: The row of the pixel.
    :param col: The column of the pixel.
    :param neighborhood_size: The requested neighborhood size.
    :return: An array of ints, sized according to the maximal value of pixels in the image.
    The value at index i contains the amount of pixels in the image with the value i.
    """
    assert neighborhood_size % 2 == 1, "Neighborhood size must be odd!"
    histogram = [0 for _ in range(256)]
    neighborhood_start_position = (row - neighborhood_size // 2, col - neighborhood_size // 2)
    for row_index in range(neighborhood_start_position[0], neighborhood_start_position[0] + neighborhood_size):
        for col_index in range(neighborhood_start_position[1], neighborhood_start_position[1] + neighborhood_size):
            row_index, col_index = normalize_indexes(img.shape, row_index, col_index)
            histogram[img[row_index][col_index]] += 1
    return histogram


def calculate_value(img: numpy.ndarray, row: int, col: int, new_img: numpy.ndarray):
    """
    Calculates the value of the pixel at position (row, col) according to its local histogram map at the original image.
    Assigns the value at new_img[row][col].
    """
    neighborhood_size = 7
    histogram = get_local_histogram(img, row, col, neighborhood_size)
    histogram_equalization = get_histogram_equlization_map(histogram, neighborhood_size ** 2)
    new_img[row][col] = histogram_equalization[img[row][col]]


def main():
    image_path = join(dirname(abspath(__file__)), "embedded_squares.jpg")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    new_img = numpy.ndarray(shape=img.shape, dtype=img.dtype)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            calculate_value(img, row, col, new_img)
    plt.imshow(new_img, cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()
