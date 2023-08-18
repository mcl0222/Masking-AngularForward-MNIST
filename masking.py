import numpy as np
from skimage.morphology import dilation, square

def masking(row_size, col_size, percentage_of_masking, binary_image, mask_size):
    # Check if input binary_image is of correct size
    if binary_image.shape[0] != row_size or binary_image.shape[1] != col_size:
        raise ValueError('Input binary_image must have dimensions equal to row_size and col_size')

    # Calculate the number of pixels to be masked
    total_pixels = row_size * col_size
    num_masked_pixels = round(total_pixels * percentage_of_masking / 100)

    # Generate random indices for masking
    rand_indices = np.random.choice(total_pixels, num_masked_pixels, replace=False)

    # Create a mask matrix
    mask = np.zeros((row_size, col_size))
    mask[np.unravel_index(rand_indices, (row_size, col_size))] = 1
    mask = dilation(mask, square(mask_size))

    # Apply the mask to the input binary image
    masked_image = binary_image.copy()
    masked_image[mask == 1] = 0

    return masked_image

# Example usage:
# binary_image = np.random.randint(2, size=(10, 10))  # Random binary image of size 10x10
# masked_img = masking(10, 10, 20, binary_image, 3)
