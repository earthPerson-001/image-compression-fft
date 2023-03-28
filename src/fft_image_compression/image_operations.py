import cv2
import numpy

import logging

import pathlib
import sys
import os

# exposing the utils folder
SRC_DIR = pathlib.Path(__file__).parents[1]

# adding src to python path
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))  # add SRC_DIR to PATH
SRC_DIR = pathlib.Path(os.path.relpath(SRC_DIR, pathlib.Path.cwd()))  # relative

from utils.compress import compress, CompressionProfiler


# some constants
IMAGES_FOLDER = pathlib.Path(pathlib.Path(__file__).parents[2]).joinpath("images")
LOG_LEVEL = logging.DEBUG


def initialize_logging(level: str | int | None) -> logging.Logger:
    """
    Creates new logger and sets the level to the given level
    """
    logger = logging.getLogger("image_compression_logger")
    logging.basicConfig(level=LOG_LEVEL if level is None else level)

    return logger


def start():
    logger = initialize_logging(LOG_LEVEL)

    # image name and path
    image_name = f"640316.jpg"
    img_path = IMAGES_FOLDER.joinpath(image_name)

    # loading the image
    logger.debug(f"Reading Image: {img_path}")
    img: cv2.Mat = cv2.imread(str(img_path))  # openCV uses BGR format

    if img is None:
        sys.exit("The Image couldn't be read.")

    original_images = ["BGR", "B", "G", "R"]
    offset_x = img.shape[1]
    offset_y = img.shape[0] + 20

    # the windows to display the original image/s
    for i, image_name in enumerate(original_images):
        cv2.namedWindow(
            f"original_image_{image_name}",
            cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED
        )
        cv2.moveWindow(f"original_image_{image_name}", i * offset_x, 10)

    # the windows to display the compressed image/s
    for i, image_name in enumerate(original_images):
        cv2.namedWindow(
            f"compressed_image_{image_name}",
            cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED
        )
        cv2.moveWindow(f"compressed_image_{image_name}",  i * offset_x, offset_y)

    main_profiler = CompressionProfiler(0)

    with main_profiler:
        # compression
        logger.debug("Starting Image compression.")
        compressed_img = compress(img)
        logger.debug("Finished Image compression")

    logger.debug(f"Compression took {main_profiler.t} seconds.")

    while True:
        # the windows to display the original image/s
        for i, image_name in enumerate(original_images):
            image_to_show = numpy.zeros((img.shape), dtype=img.dtype)

            if i == 0:  # showing the original image
                image_to_show = img
            else:  # copying only one channel in case of separate images
                image_to_show[:, :, i - 1] = img[:, :, i - 1]

            cv2.imshow(f"original_image_{image_name}", image_to_show)

        # the windows to display the compressed image/s
        for i, image_name in enumerate(original_images):
            image_to_show = numpy.zeros(
                (compressed_img.shape), dtype=compressed_img.dtype
            )

            if i == 0:  # showing the original image
                image_to_show = compressed_img
            else:  # copying only one channel in case of separate images
                image_to_show[:, :, i - 1] = compressed_img[:, :, i - 1]

            cv2.imshow(f"compressed_image_{image_name}", image_to_show)

        # showing the difference
        diff_img = numpy.abs(numpy.subtract(img,compressed_img))
        cv2.namedWindow("Diff Img", cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
        cv2.imshow("Diff Img", diff_img)

        # pressing "q" will close the window
        k = cv2.waitKey() & ord("q")

        if k:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    start()
