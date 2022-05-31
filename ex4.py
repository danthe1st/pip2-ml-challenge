import numpy as np
import numpy.typing as npt


def ex4(image_array: npt.NDArray[np.float64],
        offset: (int, int),
        spacing: (int, int)) -> (npt.NDArray[np.float64], npt.NDArray[np.int], npt.NDArray[np.float64]):
    if not isinstance(image_array, np.ndarray):
        raise TypeError()
    if len(image_array.shape) != 3 or image_array.shape[2] != 3:
        raise NotImplementedError()

    if not isinstance(offset, tuple) or not isinstance(spacing, tuple):
        raise ValueError()


    offset = (int(offset[0]), int(offset[1]))  # tries to convert to int and raise ValueError if not possible
    spacing = (int(spacing[0]), int(spacing[1]))

    offset=(offset[1],offset[0])
    spacing=(spacing[1],spacing[0])

    check_bounds(offset[0], 0, 32)
    check_bounds(offset[1], 0, 32)
    check_bounds(spacing[0], 2, 8)
    check_bounds(spacing[1], 2, 8)

    known_pixel_count = len(range(offset[0], image_array.shape[0], spacing[0]))*len(range(offset[1], image_array.shape[1], spacing[1]))

    if known_pixel_count < 144:
        raise ValueError()

    known_array = np.zeros((image_array.shape[0],image_array.shape[1],3), dtype=image_array.dtype)
    target_array = np.zeros((3, image_array.shape[0] * image_array.shape[1] - known_pixel_count),
                            dtype=image_array.dtype)

    for x in range(offset[0], image_array.shape[0], spacing[0]):
        for y in range(offset[1], image_array.shape[1], spacing[1]):
            known_array[x, y,:] = (1, 1, 1)

    target_index = 0
    for x in range(0, image_array.shape[0]):
        for y in range(0, image_array.shape[1]):
            if known_array[x, y, 0] == 0:
                for z in range(0, 3):
                    target_array[z,target_index] = image_array[x, y, z]
                target_index = target_index + 1

    input_array = image_array * known_array

    return np.transpose(input_array, (2, 0, 1)), np.transpose(known_array,(2,0,1)), target_array.flatten()


def check_bounds(to_check, min, max):
    if to_check < min or to_check > max:
        raise ValueError()
