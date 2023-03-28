import numpy
from scipy.fft import fft2, ifft2


def fft_2d(one_channel_img: numpy.ndarray):
    """
    Performs fft on one channel image of shape(height, width, 1)

    Returns numpy array of the discrete fourier transform

    """
    # axes=(0, 1) # along height and width
    return numpy.asarray(fft2(one_channel_img))  # axes=axes


def ifft_2d(coefficients: numpy.ndarray, output_shape = None):
    """
    Performs inverse fast fourier transform on given coefficients.

    Returns a numpy ndarray of given shape
    """

    return numpy.asarray(ifft2(coefficients))  # output_shape=output_shape
