
"""
Challenge 8 experiment runner.

Run spread-spectrum (SS) and STDM watermarking experiments.

The 8192-bit parameter values are the reference values calibrated
for approximately 40 dB PSNR:

    gamma_8192 = 14.375
    Delta_8192 = 49.25

For a payload of N bits, both parameters are scaled by:

    sqrt(8192 / N)

This approximately maintains the total embedding distortion, and
therefore the PSNR, across different payload lengths.
"""

import csv
from pathlib import Path

import numpy as np

from watermarking import (
    IMAGE_PATH,
    OUTPUT_DIR,
    PAYLOAD_LENGTHS,
    JPEG_QUALITY,
    GAMMA,
    DELTA,
    EMBEDDING_KEY,
    SEQUENCE_KEY,
    MESSAGE_SEED,
    load_image,
    image_to_dct,
    extract_selected_coefficients,
    make_tiles,
    generate_spreading_sequences,
    generate_message,
    embed_ss,
    embed_stdm,
    reconstruct_image,
    extract_from_image,
    jpeg_attack,
    psnr,
    ber,
    save_image,
)

from detectors import detect_ss, detect_stdm


# ---------------------------------------------------------------------
# Reference parameters
# ---------------------------------------------------------------------

# GAMMA and DELTA in watermarking.py are the values calibrated
# for the reference payload of 8192 bits.
REFERENCE_PAYLOAD_BITS = 8192

GAMMA_8192 = GAMMA
DELTA_8192 = DELTA


def scaled_parameters(number_of_bits):
    """
    Calculate SS and STDM parameters for the requested payload length.

    The square-root scaling approximately keeps total embedding
    distortion constant when the number of embedded bits changes.
    """
    if number_of_bits <= 0:
        raise ValueError("The payload length must be positive.")

    scale = np.sqrt(REFERENCE_PAYLOAD_BITS / number_of_bits)

    gamma = GAMMA_8192 * scale
    Delta = DELTA_8192 * scale

    return gamma, Delta


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------

def run_experiment():

    # Check that the input image exists.
    if not IMAGE_PATH.is_file():
        raise FileNotFoundError(
            f"Could not find {IMAGE_PATH}. "
            "Copy the input image into the data folder "
            "or check IMAGE_PATH in watermarking.py."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load the original image and calculate its DCT coefficients.
    image = load_image(IMAGE_PATH)
    original_dct = image_to_dct(image)

    # Pool the selected mid-frequency coefficients from the whole image.
    coefficients = extract_selected_coefficients(original_dct)

    print("Image:", IMAGE_PATH.name)
    print("Image dimensions:", image.shape)
    print("Number of selected coefficients:", len(coefficients))
    print("Reference payload:", REFERENCE_PAYLOAD_BITS, "bits")
    print("Reference gamma:", GAMMA_8192)
    print("Reference Delta:", DELTA_8192)
    print("JPEG quality:", JPEG_QUALITY)

    rows = []

    # Run the experiments for each payload length.
    for n_bits in PAYLOAD_LENGTHS:

        # Calculate the parameters for this payload length.
        gamma, Delta = scaled_parameters(n_bits)

        # Display the parameters once for this payload.
        print()
        print("=" * 58)
        print(f"Payload: {n_bits} bits")
        print(f"gamma = {gamma:.3f}    Delta = {Delta:.3f}")
        print("=" * 58)

        # Generate the coefficient tiles, spreading sequences and message.
        # Both watermarking methods use the same allocation and message.
        tiles = make_tiles(
            len(coefficients),
            n_bits,
            EMBEDDING_KEY
        )

        sequences = generate_spreading_sequences(
            tiles,
            SEQUENCE_KEY
        )

        message = generate_message(
            n_bits,
            MESSAGE_SEED
        )

        # Display the results table header.
        print(
            f"{'Method':<8}"
            f"{'Clean PSNR':>12}"
            f"{'Clean BER':>12}"
            f"{'JPEG PSNR':>12}"
            f"{'JPEG BER':>12}"
        )
        print("-" * 56)

        # -------------------------------------------------------------
        # Spread-spectrum and STDM experiments
        # -------------------------------------------------------------

        for method_name in ("SS", "STDM"):

            if method_name == "SS":

                # Embed the message using spread spectrum.
                marked_coefficients = embed_ss(
                    coefficients,
                    tiles,
                    sequences,
                    message,
                    gamma
                )

            else:

                # Embed the message using STDM.
                marked_coefficients = embed_stdm(
                    coefficients,
                    tiles,
                    sequences,
                    message,
                    Delta
                )

            # Reconstruct the watermarked image.
            marked_image = reconstruct_image(
                marked_coefficients,
                original_dct
            )

            # Extract the coefficients again from the reconstructed image.
            # This includes the effects of image rounding and clipping.
            clean_coefficients = extract_from_image(marked_image)

            # Detect the message from the clean watermarked image.
            if method_name == "SS":

                clean_detected = detect_ss(
                    clean_coefficients,
                    tiles,
                    sequences
                )

            else:

                clean_detected = detect_stdm(
                    clean_coefficients,
                    tiles,
                    sequences,
                    Delta
                )

            # ---------------------------------------------------------
            # JPEG attack
            # ---------------------------------------------------------

            attacked_image = jpeg_attack(
                marked_image,
                JPEG_QUALITY
            )

            attacked_coefficients = extract_from_image(
                attacked_image
            )

            # Detect the message after the JPEG attack.
            if method_name == "SS":

                attacked_detected = detect_ss(
                    attacked_coefficients,
                    tiles,
                    sequences
                )

            else:

                attacked_detected = detect_stdm(
                    attacked_coefficients,
                    tiles,
                    sequences,
                    Delta
                )

            # ---------------------------------------------------------
            # Quality and detection performance
            # ---------------------------------------------------------

            clean_psnr = psnr(
                image,
                marked_image
            )

            clean_ber = ber(
                message,
                clean_detected
            )

            attacked_psnr = psnr(
                image,
                attacked_image
            )

            attacked_ber = ber(
                message,
                attacked_detected
            )

            # ---------------------------------------------------------
            # Save the watermarked images
            # ---------------------------------------------------------

            prefix = f"{method_name.lower()}_{n_bits}"

            save_image(
                marked_image,
                OUTPUT_DIR / f"{prefix}_clean.png"
            )

            save_image(
                attacked_image,
                OUTPUT_DIR /
                f"{prefix}_jpeg_q{JPEG_QUALITY}.png"
            )

            # ---------------------------------------------------------
            # Store the results
            # ---------------------------------------------------------

            row = {
                "payload_bits": n_bits,
                "method": method_name,
                "gamma": gamma if method_name == "SS" else "",
                "Delta": Delta if method_name == "STDM" else "",
                "jpeg_quality": JPEG_QUALITY,
                "clean_psnr_db": clean_psnr,
                "clean_ber": clean_ber,
                "jpeg_psnr_db": attacked_psnr,
                "jpeg_ber": attacked_ber,
            }

            rows.append(row)

            # Print the results for this experiment.
            print(
                f"{method_name:<8}"
                f"{clean_psnr:>12.3f}"
                f"{clean_ber:>12.6f}"
                f"{attacked_psnr:>12.3f}"
                f"{attacked_ber:>12.6f}"
            )

    # -----------------------------------------------------------------
    # Save all results to a CSV file
    # -----------------------------------------------------------------

    csv_path = OUTPUT_DIR / "results.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 58)
    print("Experiments completed.")
    print("Detailed results saved to:", csv_path)
    print("Watermarked images saved to:", OUTPUT_DIR.resolve())
    print("=" * 58)


# ---------------------------------------------------------------------
# Program entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    run_experiment()
