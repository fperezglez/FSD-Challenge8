
"""
Student detector functions.

Complete detect_ss() and detect_stdm().

Use the same tiles and normalized spreading sequences p_i
that were generated for embedding.
"""

import numpy as np

from watermarking import DELTA, distance_to_coset


def detect_ss(coefficients, tiles, sequences):
    """
    Estimate each Spread Spectrum bit from the correlation
    between the received tile vector and its spreading sequence.

    Return one bit (-1 or +1) per tile, in message order.
    """
    detected = []

    for tile, p_i in zip(tiles, sequences):

        # TODO 1: Extract the received coefficient vector r_i.
        # TODO 2: Compute the correlation r_i^T p_i.
        # TODO 3: Decide +1 if the correlation is >= 0,
        #         and -1 otherwise.

        raise NotImplementedError("Complete detect_ss().")

    return np.asarray(detected, dtype=int)


def detect_stdm(
    coefficients,
    tiles,
    sequences,
    Delta=DELTA
):
    """
    Estimate each STDM bit by comparing distances to the
    two previously defined quantization cosets.

    Return one bit (-1 or +1) per tile, in message order.
    """
    detected = []

    for tile, p_i in zip(tiles, sequences):

        # TODO 1: Extract the received coefficient vector r_i.
        # TODO 2: Compute the received projection:
        #         u_hat_i = r_i^T p_i.
        # TODO 3: Compute the distance to the -1 coset using
        #         distance_to_coset().
        # TODO 4: Compute the distance to the +1 coset.
        # TODO 5: Decide -1 if d_minus <= d_plus,
        #         and +1 otherwise.

        raise NotImplementedError("Complete detect_stdm().")

    return np.asarray(detected, dtype=int)


