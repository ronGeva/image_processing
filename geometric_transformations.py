import cv2
import matplotlib.pyplot as plt
from math import sin, cos, pi
import numpy
import sys


def rotate(img: numpy.ndarray, theta: float):
    padding_needed = int(max(img.shape) * 1.5)
    transformation_matrix = numpy.float32([[cos(theta), sin(theta), padding_needed // 2],
                                           [-sin(theta), cos(theta), padding_needed // 2]])
    res = cv2.warpAffine(img, transformation_matrix, (img.shape[0] + padding_needed, img.shape[1] + padding_needed))
    plt.imshow(res, cmap='gray')
    plt.show()


def scale(img: numpy.ndarray, scale: float):
    transformation_matrix = numpy.float32([[scale, 0, 0],
                                           [0, scale, 0]])
    res = cv2.warpAffine(img, transformation_matrix, (img.shape[0] * scale, img.shape[1] * scale))
    plt.imshow(res, cmap='gray')
    plt.show()


def main(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    plt.imshow(img, cmap='gray')
    plt.show()
    rotate(img, pi / 3)


if __name__ == '__main__':
    main(sys.argv[1])
