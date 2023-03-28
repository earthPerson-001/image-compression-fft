import numpy
import cv2


def display_all(img: numpy.ndarray, compressed_img: numpy.ndarray):
    """
    Display images and respective channels
    """

    max_height = max(img.shape[0], compressed_img.shape[0])
    max_width = max(img.shape[1], compressed_img.shape[1])

    cv2.namedWindow("Test Window", cv2.WINDOW_NORMAL | cv2.WINDOW_FULLSCREEN)
    rect = cv2.getWindowImageRect("Test Window")
    print(rect)
    cv2.destroyWindow("Test Window")
    input("Waiting...")

    image_height = 2 * max_height  # multiple of maximum heights
    image_width = (
        max(img.shape[2], compressed_img.shape[2]) * max_width
    )  # multiple of maximum channels

    # creating a bigger image to place all the images
    height = image_height
    width = image_width

    whole_img = numpy.zeros((height, width, img.shape[2]), dtype=img.dtype)
    horizontal_padding = int((whole_img.shape[1] - 4 * img.shape[1]) / 3)
    vertical_padding = whole_img.shape[0] - (img.shape[0] + compressed_img.shape[0])

    # the complete original image
    whole_img[: img.shape[0], : img.shape[1], : whole_img.shape[2]] = img[
        :, :, : whole_img.shape[2]
    ]

    # individual channels of original image
    for channel in range(img.shape[2]):
        whole_img[
            : img.shape[0],
            channel * horizontal_padding
            + (channel + 1) * img.shape[1] : horizontal_padding * (channel + 1)
            + (channel + 1) * img.shape[1],
            channel,
        ] = 255
        whole_img[
            : img.shape[0],
            (channel + 1) * horizontal_padding
            + (channel + 1) * img.shape[1] : (channel + 1) * horizontal_padding
            + (channel + 2) * img.shape[1],
            channel,
        ] = img[:, :, channel]

    # vertical gap
    whole_img[img.shape[0] : img.shape[0] + vertical_padding, :, :] = 255

    starting_height = img.shape[0] + vertical_padding
    # the complete compressed image
    whole_img[
        starting_height : starting_height + compressed_img.shape[0],
        : compressed_img.shape[1],
        : whole_img.shape[2],
    ] = compressed_img[:, :, : whole_img.shape[2]]

    # individual channels of compressed image
    for channel in range(compressed_img.shape[2]):
        whole_img[
            starting_height : starting_height + compressed_img.shape[0],
            channel * horizontal_padding
            + (channel + 1)
            * compressed_img.shape[1] : horizontal_padding
            * (channel + 1)
            + (channel + 1) * compressed_img.shape[1],
            channel,
        ] = 255
        whole_img[
            starting_height : starting_height + compressed_img.shape[0],
            horizontal_padding
            + (channel + 1)
            * compressed_img.shape[1] : horizontal_padding
            * (channel + 1)
            + (channel + 2) * compressed_img.shape[1],
            channel,
        ] = compressed_img[:, :, channel]

    cv2.namedWindow("Main Window", cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)

    while True:
        cv2.imshow("Main Window", whole_img)

        # pressing "q" will close the window
        k = cv2.waitKey() & ord("q")

        if k:
            break


if __name__ == "__main__":
    img_1 = numpy.random.randint(0, 255 + 1, size=(800, 800, 3))
    img_2 = numpy.random.randint(0, 255 + 1, size=(800, 800, 3))

    display_all(img_1, img_2)
