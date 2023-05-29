import cv2

import logging

import pathlib
import sys
import os

from urllib import request

import numpy as np

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

    USE_FROM_URL = True
    GRAYSCALE_ONLY = True

    if USE_FROM_URL:
        image_name = "2012-landscapes-c2a9-christopher-martin-93441.jpg"
        req = request.urlopen(f'https://chrismartinphotography.files.wordpress.com/2013/01/{image_name}')
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, -1) # 'Load it as it is'
    else:
        # image name and path
        image_name = f"mt_everest.jpg"
        img_path = IMAGES_FOLDER.joinpath(image_name)

        # loading the image
        logger.debug(f"Reading Image: {img_path}")
        img: cv2.Mat = cv2.imread(str(img_path))  # openCV uses BGR format

    if img is None:
        sys.exit("The Image couldn't be read.")

    if GRAYSCALE_ONLY:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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

        image_extension = image_name.rsplit(".")[-1]
        image_name_only = image_name.split(".")[0]
        image_save_path = pathlib.Path(
            str(
                IMAGES_FOLDER.joinpath(
                    f"{image_name_only}_compression_{compression_percentage}.{image_extension}"
                )
            )
        )

        exists = "exists" if image_save_path.exists() else "doesn't exist"
        print(f"Saving image to {image_save_path}, \n the path {exists}")
        cv2.imwrite(image_save_path.__str__(), img=compressed_img, params=[cv2.IMWRITE_JPEG_QUALITY, 55])

        # runs until key press events (q or n)
        cont = display_all(images, texts=texts)

        if not cont:
            logger.info("Exited as per user request.")
            return


if __name__ == "__main__":
    start()
