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

from utils.compress import compress
from utils.image_display import display_all


# some constants
IMAGES_FOLDER = pathlib.Path(pathlib.Path(__file__).parents[2]).joinpath("images")
LOG_LEVEL = logging.DEBUG


def initialize_logging(level: str | int = LOG_LEVEL) -> logging.Logger:
    """
    Creates new logger and sets the level to the given level
    """
    logger = logging.getLogger("image_compression_logger")
    logging.basicConfig(level=level)

    return logger


def start():
    logger = initialize_logging(LOG_LEVEL)

    # image name and path
    image_name = f"mt_everest.jpg"
    img_path = IMAGES_FOLDER.joinpath(image_name)

    # loading the image
    logger.debug(f"Reading Image: {img_path}")
    img: cv2.Mat = cv2.imread(str(img_path))  # openCV uses BGR format

    if img is None:
        sys.exit("The Image couldn't be read.")

    images = [img]
    texts = ["Original Image"]

    if not display_all(images, texts=texts):
        return

    for compression_percentage in range(50, 100, 10):

        compression_threshold = compression_percentage * 0.01
        # compression
        compressed_img = compress(img, compression_threshold)

        images.append(compressed_img)
        texts.append(f"Compression {compression_percentage}%.")

        # runs until key press events (q or n)
        cont = display_all(images, texts=texts)
           
        if not cont:
            logger.info("Exited as per user request.")
            return

if __name__ == "__main__":
    start()
