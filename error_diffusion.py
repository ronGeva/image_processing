import sys
from os.path import isfile, basename, dirname, splitext, join
import cv2
import matplotlib.pyplot as plt
import numpy


def parse_image_path(args: list):
    """
    Extracts the image path from the command line arguments.
    :param args: The command line arguments passed to the program.
    :return: The image path in case the parse was successful, otherwise - None.
    """
    if len(args) < 2:
        print("Usage: python ex3.py <colored_image_path>")
        return None
    path = args[1]
    if not isfile(path):
        print("Path {path} isn't a valid file path. Please use a valid path.".format(path=path))
        return None
    return path


def add_diff_to_neighbors(img: numpy.ndarray, i: int, j: int, k: int, diff: int):
    """
    Changes the right, bottom, right-bottom neighbors of pixel [i][j] to distribute the error among them, resulting
    in a more accurate image.
    """
    if i < len(img) - 1:
        img[i + 1][j][k] += (3 / 8) * diff
    if j < len(img[i]) - 1:
        img[i][j + 1][k] += (3 / 8) * diff
    if i < len(img) - 1 and j < len(img[i]) - 1:
        img[i + 1][j + 1][k] += (1 / 4) * diff


def get_color_threshold(dtype: numpy.dtype):
    return pow(2, 8 * dtype.itemsize)  # k bits of data can contain 2^(8k) different values


def error_diffusion_encode(input_image: numpy.ndarray):
    """
    Encodes the input image using error-diffusion algorithm by matching each RGB value in each pixel to one of 8
     possible values, thus using only 9 bits for each pixel.
    We use an int-sized array to temporarily store the new values of the pixel array since during the algorithm some
    pixels' colors might exceed the 0-255 range before we'll adjust them to the proper color level (for example, a pixel
    with the Red color value 252 might exceed 255 if its left neighbor's diff will be 20).
    :param input_image: The input image to encode.
    :return: A temporary array containing the encoded image with each pixel represented by 3 integers (for each color)
    each containing one of 8 possible values, encoded using the error diffusion algorithm.
    """
    new_image = numpy.zeros(input_image.shape, dtype=int)
    new_image[:] = input_image

    color_threshold = get_color_threshold(input_image.dtype)
    # first color level is 0, then we need 7 more possible levels to achieve 8 total color levels
    color_level_size = color_threshold // 7
    for i in range(len(new_image)):
        for j in range(len(new_image[i])):
            for k in range(len(new_image[i][j])):
                original_value = new_image[i][j][k]
                new_image[i][j][k] = (new_image[i][j][k] // color_level_size) * color_level_size
                diff = original_value - new_image[i][j][k]
                add_diff_to_neighbors(new_image, i, j, k, diff)
    return new_image


def write_encoded_image(input_path: str, encoded_image: numpy.ndarray):
    """
    Copies the encoded image into an uint8 sized array (since each color only contains 8 possible color values) then
    writes the new array into a png image.
    """
    output_image = numpy.zeros(encoded_image.shape, dtype=numpy.dtype('uint8'))
    output_image[:] = encoded_image
    basename_no_suffix = splitext(basename(input_path))[0]
    output_path = join(dirname(input_path), "encoded_{path}.png".format(path=basename_no_suffix))
    cv2.imwrite(output_path, output_image)


def main(args: list):
    path = parse_image_path(args)
    if path is None:
        return
    input_image = cv2.imread(path)
    plt.imshow(input_image)
    plt.show()

    encoded_image = error_diffusion_encode(input_image)
    plt.imshow(encoded_image)
    plt.show()

    write_encoded_image(path, encoded_image)


if __name__ == '__main__':
    main(sys.argv)
