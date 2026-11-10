# Challenge 8 --- Protecting the prototype

**Multimedia Security: Robust Digital Watermarking**

This repository contains the material and Python code for Challenge 8.
The exercise introduces two transform-domain watermarking
techniques---Spread Spectrum (SS) and Spread Transform Dither Modulation / STDM---and
compares their detection performance before and after JPEG compression.

## Objectives

Students will: - understand watermark embedding in selected DCT
coefficients; - implement SS and STDM detectors; - evaluate image
quality using PSNR; - evaluate watermark recovery using bit error rate
(BER); - compare clean and JPEG-attacked detection; and - explain how
the embedding parameters scale with payload length.

## Repository contents

The project is organised into student and instructor versions:

``` text
.
├── README.md
├── watermarking.py
├── detectors.py
├── run_experiments.py
├── requirements.txt
└── data/
     └── t046.png

```

detectors.py contains the detector functions to be completed. The
instructor version contains reference implementations.

The input image may be distributed separately. Place the image specified
by your instructor in the relevant `data/` directory. Check `IMAGE_PATH`
in `watermarking.py` and make sure the filename matches your image. For
example:

``` python
IMAGE_PATH = Path(__file__).resolve().parent / "data" / "t046.png"
```

Do not rename an image just to match this example unless it is the
intended image.

## Requirements

The code uses Python and the following packages: - NumPy - SciPy -
Pillow

Install the dependencies from the relevant project directory:

``` bash
python -m pip install -r requirements.txt
```

Alternatively:

``` bash
python -m pip install numpy scipy pillow
```

## Running the experiment

1.  Place the supplied image in the `data/` directory.
2.  Complete the two detector functions in `detectors.py` (student
    version).
3.  Check the `IMAGE_PATH` setting in `watermarking.py`.
4.  From the directory containing `run_experiments.py`, execute:

``` bash
python run_experiments.py
```

The script runs both methods for payload lengths of 2048, 4096 and 8192
bits in one execution. It prints the results and saves the generated
images and a CSV file in the `results/` directory.

The reported metrics are: - **Clean PSNR:** image quality after
watermark embedding and reconstruction, before JPEG compression. -
**Clean BER:** bit error rate before JPEG compression. - **JPEG PSNR:**
image quality after JPEG compression. - **JPEG BER:** bit error rate
after JPEG compression.

The CSV file also records the parameter values used for each experiment.

## Experimental methods

### Spread Spectrum (SS)

Each message bit is embedded in a tile of selected DCT coefficients by
adding a scaled, unit-norm spreading sequence. The embedding strength is
controlled by `gamma`. Detection correlates the received tile with its
spreading sequence and decides the bit from the sign of the correlation.

### Spread Transform Dither Modulation / STDM

For each tile, the host coefficient vector is projected onto a unit-norm
spreading sequence. The projection is quantized to the coset associated
with the message bit, with quantization step `Delta`. Detection computes
the received projection and chooses the closest of the two
bit-associated cosets.

The challenge handout gives the mathematical definitions of the
spreading sequences, projections, cosets, embedding rules and detection
rules.

## Payload-dependent parameter scaling

The 8192-bit reference parameters are:

\[ `\gamma`{=tex}*{8192}=14.375, `\qquad`{=tex}
`\Delta`{=tex}*{8192}=49.25. \]

For a payload of (N) bits, the script applies:

\[ `\gamma`{=tex}(N)=`\gamma`{=tex}*{8192}`\sqrt{\frac{8192}{N}}`{=tex},
`\qquad`{=tex}
`\Delta`{=tex}(N)=`\Delta`{=tex}*{8192}`\sqrt{\frac{8192}{N}}`{=tex}. \]

This rule approximately maintains the total embedding distortion as the
payload length changes. It is an energy-based approximation, not a
guarantee of exactly 40 dB PSNR for every image and payload. The final
task in the handout asks students to explain the relationship between
this scaling rule, embedding distortion and PSNR.

## Reproducibility

Fixed seeds are used for message generation, spreading-sequence
generation and coefficient-tile selection. With the same input image,
code, package versions and parameter settings, results should be
reproducible.

JPEG compression uses the quality setting defined in `watermarking.py`.

## Further information

Refer to the challenge handout for the full scenario, mathematical
description, tasks and submission instructions. The instructor version
includes reference detector implementations and suggested solutions.
