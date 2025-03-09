import numpy as np
from PIL import Image

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]


def convert_rgb_to_rgba_with_resize(
    rgb_image_path: str,
    npy_mask_path: str,
    output_path: str,
    max_dimension: int = 512
) -> Image.Image:
    """
    Load an RGB image from 'rgb_image_path', resize its largest dimension to max_dimension,
    load and resize a 2D mask from 'npy_mask_path' using the same scale factor,
    combine them into an RGBA image (with the mask as the alpha channel),
    and save the resulting image to 'output_path'.

    Parameters
    ----------
    rgb_image_path : str
        Path to the RGB image file.
    npy_mask_path : str
        Path to the .npy mask file (2D NumPy array).
    output_path : str
        Where to save the resulting RGBA image.
    max_dimension : int
        The largest dimension (width or height) for the resized image. Default is 512.

    Returns
    -------
    Image.Image
        The RGBA image (PIL Image object) after resizing and applying the alpha mask.
    """

    # 1) Load the RGB image
    rgb_img = Image.open(rgb_image_path).convert('RGB')
    orig_width, orig_height = rgb_img.size
    
    # Determine the scaling factor so that the largest dimension = max_dimension
    largest_dim = max(orig_width, orig_height)
    if largest_dim > 0:
        scale = max_dimension / largest_dim
    else:
        raise ValueError("Invalid image dimensions (width or height is 0).")

    # Compute new width and height (must be integers for PIL resize)
    new_width = int(round(orig_width * scale))
    new_height = int(round(orig_height * scale))

    mask_np = (1-np.load(npy_mask_path))*255
    new_height, new_width = mask_np.shape

    # Resize the RGB image
    # Using LANCZOS for high-quality downsampling
    rgb_img_resized = rgb_img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # 2) Load the mask from the .npy file
    
    # Ensure the mask is 2D
    if mask_np.ndim != 2:
        raise ValueError(f"Mask array must be 2D, but got shape {mask_np.shape}")
    
    # Convert mask to uint8 (PIL expects 8-bit for alpha channel)
    if mask_np.dtype != np.uint8:
        mask_np = mask_np.astype(np.uint8)

    # 4) Combine the resized RGB and mask into an RGBA array
    rgb_array_resized = np.array(rgb_img_resized)  # shape: (new_height, new_width, 3)
    rgba_array = np.dstack((rgb_array_resized, mask_np))  # shape: (..., 4)

    # 5) Create an RGBA PIL image and save
    rgba_img = Image.fromarray(rgba_array, mode='RGBA')
    rgba_img.save(output_path)

    return



def radical_inverse(base, n):
    val = 0
    inv_base = 1.0 / base
    inv_base_n = inv_base
    while n > 0:
        digit = n % base
        val += digit * inv_base_n
        n //= base
        inv_base_n *= inv_base
    return val

def halton_sequence(dim, n):
    return [radical_inverse(PRIMES[dim], n) for dim in range(dim)]

def hammersley_sequence(dim, n, num_samples):
    return [n / num_samples] + halton_sequence(dim - 1, n)

def sphere_hammersley_sequence(n, num_samples, offset=(0, 0), remap=False):
    u, v = hammersley_sequence(2, n, num_samples)
    u += offset[0] / num_samples
    v += offset[1]
    if remap:
        u = 2 * u if u < 0.25 else 2 / 3 * u + 1 / 3
    theta = np.arccos(1 - 2 * u) - np.pi / 2
    phi = v * 2 * np.pi
    return [phi, theta]

if __name__=="__main__":
    convert_rgb_to_rgba_with_resize("/data/ziqi/data/3dwild/vase/vase3.png", "/data/ziqi/data/3dwild/vase/vase3-mask.npy", "/data/ziqi/data/3dwild/vase/vase3_rgba.png")