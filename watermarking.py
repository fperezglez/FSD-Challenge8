
"""
Shared utilities for the Spread Spectrum and STDM watermarking experiment.

The image is transformed using non-overlapping 8x8 DCT blocks. The twelve
selected mid-frequency AC coefficients from every block are pooled into
one global vector. A seeded permutation partitions that vector into disjoint
tiles, one tile per payload bit.

Each tile therefore contains coefficients distributed throughout the image;
payload bits are NOT assigned to spatial image blocks.
"""

from pathlib import Path
from io import BytesIO

import numpy as np
from PIL import Image
from scipy.fft import dctn, idctn


# ---------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------

IMAGE_PATH = Path(__file__).resolve().parent / "data" / "t064.png"
OUTPUT_DIR = Path(__file__).resolve().parent / "results"

PAYLOAD_LENGTHS = (2048, 4096, 8192)

BLOCK_SIZE = 8
JPEG_QUALITY = 60

GAMMA = 14.375
DELTA = 49.25

EMBEDDING_KEY = 12345
SEQUENCE_KEY = 54321
MESSAGE_SEED = 9876

# Twelve mid-frequency AC coefficient positions in each 8x8 DCT block.
MID_FREQ = (
    (1, 2), (2, 1), (2, 2), (1, 3),
    (3, 1), (2, 3), (3, 2), (1, 4),
    (4, 1), (3, 3), (2, 4), (4, 2),
)


# ---------------------------------------------------------------------
# Image utilities and quality metrics
# ---------------------------------------------------------------------

def load_image(path):
    """Load an image as an 8-bit grayscale array represented by float64."""
    return np.asarray(
        Image.open(path).convert("L"),
        dtype=np.float64
    )


def save_image(array, path):
    """Round, clip and save an image as an 8-bit grayscale PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    array_uint8 = np.clip(
        np.rint(array), 0, 255
    ).astype(np.uint8)

    Image.fromarray(array_uint8).save(path)


def psnr(original, processed):
    """Return peak signal-to-noise ratio in dB."""
    mse = np.mean((original - processed) ** 2)

    if mse == 0:
        return float("inf")

    return float(10.0 * np.log10((255.0 ** 2) / mse))


def ber(original, detected):
    """Return the bit error rate."""
    original = np.asarray(original)
    detected = np.asarray(detected)

    if original.shape != detected.shape:
        raise ValueError(
            "Original and detected messages must have equal shapes."
        )

    return float(np.mean(original != detected))


def jpeg_attack(image, quality=JPEG_QUALITY):
    """Encode and decode an image as JPEG at the requested quality."""
    buffer = BytesIO()

    pil_image = Image.fromarray(
        np.clip(np.rint(image), 0, 255).astype(np.uint8)
    )

    pil_image.save(
        buffer,
        format="JPEG",
        quality=quality
    )

    buffer.seek(0)

    return np.asarray(
        Image.open(buffer).convert("L"),
        dtype=np.float64
    )


# ---------------------------------------------------------------------
# 8x8 block DCT and inverse DCT
# ---------------------------------------------------------------------

def image_to_dct(image):
    """Apply an orthonormal 2-D DCT to each non-overlapping 8x8 block."""
    image = np.asarray(image, dtype=np.float64)

    height, width = image.shape

    if height % BLOCK_SIZE or width % BLOCK_SIZE:
        raise ValueError(
            "Image height and width must be multiples of 8."
        )

    coeffs = np.empty(
        (
            height // BLOCK_SIZE,
            width // BLOCK_SIZE,
            BLOCK_SIZE,
            BLOCK_SIZE
        ),
        dtype=np.float64
    )

    for by in range(height // BLOCK_SIZE):
        for bx in range(width // BLOCK_SIZE):

            block = image[
                by * BLOCK_SIZE:(by + 1) * BLOCK_SIZE,
                bx * BLOCK_SIZE:(bx + 1) * BLOCK_SIZE
            ]

            coeffs[by, bx] = dctn(
                block,
                type=2,
                norm="ortho"
            )

    return coeffs


def dct_to_image(coeffs):
    """Apply the inverse orthonormal 2-D DCT to every 8x8 block."""
    nby, nbx = coeffs.shape[:2]

    image = np.empty(
        (nby * BLOCK_SIZE, nbx * BLOCK_SIZE),
        dtype=np.float64
    )

    for by in range(nby):
        for bx in range(nbx):

            block = idctn(
                coeffs[by, bx],
                type=2,
                norm="ortho"
            )

            image[
                by * BLOCK_SIZE:(by + 1) * BLOCK_SIZE,
                bx * BLOCK_SIZE:(bx + 1) * BLOCK_SIZE
            ] = block

    return np.clip(np.rint(image), 0, 255)


# ---------------------------------------------------------------------
# Pool the selected coefficients across the WHOLE image
# ---------------------------------------------------------------------

def extract_selected_coefficients(dct_coeffs):
    """Pool the twelve selected coefficients from every DCT block."""
    selected = []

    for by in range(dct_coeffs.shape[0]):
        for bx in range(dct_coeffs.shape[1]):

            for row, col in MID_FREQ:
                selected.append(
                    dct_coeffs[by, bx, row, col]
                )

    return np.asarray(selected, dtype=np.float64)


def insert_selected_coefficients(dct_coeffs, selected):
    """Insert a global selected-coefficient vector into the DCT blocks."""
    expected = (
        dct_coeffs.shape[0]
        * dct_coeffs.shape[1]
        * len(MID_FREQ)
    )

    if len(selected) != expected:
        raise ValueError(
            f"Expected {expected} selected coefficients, "
            f"got {len(selected)}."
        )

    result = dct_coeffs.copy()
    index = 0

    for by in range(result.shape[0]):
        for bx in range(result.shape[1]):

            for row, col in MID_FREQ:
                result[by, bx, row, col] = selected[index]
                index += 1

    return result


def reconstruct_image(selected, original_dct):
    """Reconstruct an image after replacing its selected DCT coefficients."""
    modified_dct = insert_selected_coefficients(
        original_dct,
        selected
    )

    return dct_to_image(modified_dct)


def extract_from_image(image):
    """Extract the global selected-coefficient vector from an image."""
    dct_coeffs = image_to_dct(image)

    return extract_selected_coefficients(dct_coeffs)


# ---------------------------------------------------------------------
# Disjoint coefficient tiles, spreading sequences and payload
# ---------------------------------------------------------------------

def make_tiles(
    number_of_coefficients,
    number_of_bits,
    key=EMBEDDING_KEY
):
    """
    Randomly permute all selected coefficients and split them into
    one disjoint tile per message bit.

    Tile sizes differ by at most one coefficient.
    """
    if number_of_bits <= 0:
        raise ValueError("The payload length must be positive.")

    if number_of_bits > number_of_coefficients:
        raise ValueError(
            f"Payload ({number_of_bits} bits) exceeds the number of "
            f"selected coefficients ({number_of_coefficients})."
        )

    rng = np.random.default_rng(key)

    permutation = rng.permutation(number_of_coefficients)

    return np.array_split(permutation, number_of_bits)


def generate_spreading_sequences(
    tiles,
    key=SEQUENCE_KEY
):
    """
    Generate one pseudorandom +/-1 sequence per tile and normalize it.

    In the handout notation:

        p_i = s_i / ||s_i||_2

    Since every element of s_i is either -1 or +1:

        ||s_i||_2 = sqrt(L_i)

    where L_i is the number of coefficients in tile i.
    Consequently, every p_i has unit Euclidean norm.
    """
    rng = np.random.default_rng(key)

    sequences = []

    for tile in tiles:

        p_i = rng.choice(
            [-1.0, +1.0],
            size=len(tile)
        )

        p_i /= np.linalg.norm(p_i)

        sequences.append(p_i)

    return sequences


def generate_message(
    number_of_bits,
    seed=MESSAGE_SEED
):
    """Generate a reproducible antipodal message with bits -1 and +1."""
    rng = np.random.default_rng(seed)

    return rng.choice(
        [-1, +1],
        size=number_of_bits
    )


# ---------------------------------------------------------------------
# Spread Spectrum embedding
# ---------------------------------------------------------------------

def embed_ss(
    coefficients,
    tiles,
    sequences,
    message,
    gamma=GAMMA
):
    """
    Embed each bit according to:

        y_i = x_i + gamma * b_i * p_i
    """
    watermarked = np.asarray(
        coefficients,
        dtype=np.float64
    ).copy()

    for b_i, tile, p_i in zip(message, tiles, sequences):

        watermarked[tile] += gamma * b_i * p_i

    return watermarked


# ---------------------------------------------------------------------
# STDM quantization and embedding
# ---------------------------------------------------------------------

def quantize_coset(value, bit, Delta=DELTA):
    """
    Return the closest point to value in the selected STDM coset.

    Q_-1 = {k*Delta - Delta/4 : k is an integer}
    Q_+1 = {k*Delta + Delta/4 : k is an integer}
    """
    if bit not in (-1, +1):
        raise ValueError("STDM bit must be -1 or +1.")

    offset = bit / 4.0

    return float(
        Delta * (
            np.round(value / Delta - offset) + offset
        )
    )


def distance_to_coset(value, bit, Delta=DELTA):
    """Return the distance to the nearest point in the selected coset."""
    nearest = quantize_coset(value, bit, Delta)

    return abs(value - nearest)


def embed_stdm(
    coefficients,
    tiles,
    sequences,
    message,
    Delta=DELTA
):
    """
    Embed each bit by modifying the projection along p_i.

    The projection u_i = x_i^T p_i is quantized to the coset
    associated with the message bit.
    """
    coefficients = np.asarray(
        coefficients,
        dtype=np.float64
    )

    watermarked = coefficients.copy()

    for b_i, tile, p_i in zip(message, tiles, sequences):

        x_i = coefficients[tile]

        u_i = np.dot(x_i, p_i)

        q_i = quantize_coset(
            u_i,
            int(b_i),
            Delta
        )

        watermarked[tile] += (q_i - u_i) * p_i

    return watermarked
