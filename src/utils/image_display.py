import numpy
import cv2


def display_all(
    img_array: list[numpy.ndarray],
    dtype: numpy.dtype | None = None,
    window_name: str = "Compression Test",
    texts: list[str] = ["Compressed"],
)-> bool:
    """
    Display images and respective channels in a single window.

    The images should contain equal channels and if no dtype if passed,
    the dtype of `img[0]` is used.

    n to continue, q to exit (destorying all the active windows)

    Returns False on termination (pressing q), can be used as value to continue
    """

    if dtype is None:
        dtype = img_array[0].dtype

    max_height = max(img.shape[0] for img in img_array)
    max_width = max(img.shape[1] for img in img_array)

    n_horizontal = (
        max(img.shape[2] for img in img_array) + 1
    )  # corresponding to maximum channel and a complete image
    n_vertical = len(img_array)  # corresponding to number of images

    image_height = n_vertical * max_height  # multiple of maximum heights
    image_width = n_horizontal * max_width  # multiple of maximum channels

    # gaps between images
    h_gap = int(0.1 * image_height)
    v_gap = int(0.1 * image_width)

    # colors for gaps (For BGR values)
    h_gap_color = [255, 255, 255]
    v_gap_color = [255, 255, 255]

    # creating a bigger image to place all the images
    height: int = int(image_height + (n_vertical) * v_gap)
    width: int = int(image_width + (n_horizontal) * h_gap)

    whole_img = numpy.zeros((height, width, n_horizontal - 1), dtype=dtype)

    # displaying the columns
    for image_number, image in enumerate(img_array):
        # displaying the rows

        # the complete image
        starting_height: int = int(image_number * (max_height + v_gap))
        whole_img[
            starting_height : starting_height + image.shape[0],
            0 : image.shape[1],
            : image.shape[2],
        ] = image

        for channel in range(image.shape[2]):
            starting_width: int = int((channel + 1) * (max_width + h_gap))

            # filling the horizontal gap
            whole_img[
                starting_height : starting_height + image.shape[0],
                starting_width - h_gap : starting_width,
            ] = h_gap_color

            start_column = starting_width
            end_column = starting_width + image.shape[1]
            whole_img[
                starting_height : starting_height + image.shape[0],
                start_column:end_column,
                channel,
            ] = image[:, :, channel]

            # last horizontal gap
            if channel == image.shape[2] - 1:
                # filling the horizontal gap
                whole_img[
                    starting_height : starting_height + image.shape[0],
                    end_column : ,
                ] = h_gap_color

        # filling the vertical gap
        
        whole_img[
            starting_height
            + image.shape[0] : starting_height
            + image.shape[0]
            + v_gap,
            :,
        ] = v_gap_color

    # adding text in the vertical gap
    for i, (image, text) in enumerate(zip(img_array, texts)):
        cv2.putText(
            whole_img,
            text,
            (10, int((i + 1)*(max_height + v_gap) - 0.5*v_gap)),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=cv2.getFontScaleFromHeight(
                cv2.FONT_HERSHEY_PLAIN, int(0.25 * v_gap), thickness=10
            ),
            color=(255, 0, 0),
            bottomLeftOrigin=False,
            thickness=10,
        )

    cv2.namedWindow(window_name, cv2.WINDOW_KEEPRATIO | cv2.WINDOW_NORMAL)

    cv2.resizeWindow(window_name, whole_img.shape[1],  whole_img.shape[0])

    while True:
        cv2.imshow(window_name, whole_img)

        # pressing "q" will close all the windows
        key = cv2.waitKey()

        if key == ord("q"):
            cv2.destroyAllWindows()
            return False

        if key == ord("n"):
            break  

    cv2.destroyWindow(window_name)
    return True


if __name__ == "__main__":
    img_1 = numpy.random.randint(0, 255 + 1, size=(800, 800, 3), dtype=numpy.uint8)
    img_2 = numpy.random.randint(0, 255 + 1, size=(800, 800, 3), dtype=numpy.uint8)
    img_3 = numpy.random.randint(0, 255 + 1, size=(800, 800, 3), dtype=numpy.uint8)


    display_all([img_1, img_2, img_3], img_1.dtype, texts=["Img1", "Img2", "Img3"])
