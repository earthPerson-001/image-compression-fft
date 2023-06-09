import numpy
import logging
import cv2

import contextlib
import time

from utils.fast_fourier_transform import fft_2d, ifft_2d

class CompressionProfile(contextlib.ContextDecorator):
    """
      Compression Profile class. Usage: @CompressionProfile() decorator or 'with CompressionProfile():' context manager
    """
    def __init__(self, t=0.0, logger: logging.Logger | None = None):
        self.t = t
        self.logger = logger

    def __enter__(self):
        self.start = self.time()
        if self.logger is not None:
            self.logger.debug("Started compression.")
        return self

    def __exit__(self, type, value, traceback):
        self.dt = self.time() - self.start  # delta-time
        if self.logger is not None:
            self.logger.debug(f"Compression took {self.dt * 1e3:.2f} ms." )
        self.t += self.dt  # accumulate dt
        if self.logger is not None:
            self.logger.debug("Finished compression.")

    def time(self):
        return time.time()

logger = logging.getLogger("compress_img")


def display_images(images: list[numpy.ndarray], window_names: list[str] | None = None):
    """
    A helper function to display multiple images in separate windows
    """

    # the windows to display the images
    for i, image in enumerate(images):
        cv2.namedWindow(
            f"Img({i})" if window_names is None else window_names[i],
            cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED,
        )

    running = True
    while running:
        for i, image in enumerate(images):
            cv2.imshow(f"Img({i})" if window_names is None else window_names[i], image)
            key = cv2.waitKey()

            if key == ord("n"):
                running = False
            if key == ord("q"):
                cv2.destroyAllWindows()
                return
    for i, image in enumerate(images):
        cv2.destroyWindow(f"Img({i})" if window_names is None else window_names[i])


@CompressionProfile(logger=logger)
def compress(original_img: numpy.ndarray, threshold=0.9) -> numpy.ndarray:
    compressed_img = numpy.zeros((original_img.shape), original_img.dtype)
    is_grayscale_image = len(compressed_img.shape) < 3

    for channel in range(1 if is_grayscale_image else compressed_img.shape[2]):
        if is_grayscale_image:
            one_channel_img = original_img
        else:
            one_channel_img = original_img[:, :, channel]

        all_coefficients = fft_2d(one_channel_img)

        # for removing the undesired coefficients
        sorted_coefficients: numpy.ndarray = numpy.sort(numpy.abs(all_coefficients.reshape(-1)))

        # since the array is sorted, we can work on that position
        threshold_index = int((threshold) * sorted_coefficients.shape[0])  # keep value
        threshold_value = sorted_coefficients[threshold_index]

        # creating a sort of mask to filter out the values less than the threshold
        mask = numpy.abs(all_coefficients) > threshold_value
        new_coefficients = all_coefficients * mask

        output_from_ifft2d = ifft_2d(new_coefficients, output_shape=original_img.shape[:2]).real

        if is_grayscale_image:
            compressed_img[:, :]= output_from_ifft2d
        else:
            compressed_img[:, :, channel] = output_from_ifft2d

        # display_images(
        #     [
        #         all_coefficients.real,
        #         new_coefficients.real,
        #         output_from_ifft2d.real,
        #         compressed_img[:, :, channel],
        #         one_channel_img,
        #         numpy.asarray(test_output).real
        #     ],
        #     [
        #         "All Coefficients",
        #         "New Coefficients",
        #         "Output from ifft2",
        #         "Compressed One Channel",
        #         "Original One Channel",
        #         "test_output"
        #     ],
        # )

    return compressed_img
