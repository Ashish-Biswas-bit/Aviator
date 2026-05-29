"""
ocr_utils.py - Optical Character Recognition utilities for extracting
numeric multiplier scores from Aviator game screenshots.

Uses Tesseract OCR (via pytesseract) with OpenCV-based image preprocessing
(grayscale, thresholding, denoising).  No PyTorch / CUDA dependency.
"""

import re
from typing import List

import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Tesseract configuration
# ---------------------------------------------------------------------------
import pytesseract

# Point pytesseract at the installed binary (adjust for your system)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

PSM_MODES = [6, 11]  # Try block mode first, then sparse text as fallback
CUSTOM_CONFIG = (
    "-c tessedit_char_whitelist=0123456789.xX "
    "--oem 3 "
    "--psm {psm}"
)

# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------


def preprocess_image(
    pil_image: Image.Image,
) -> Image.Image:
    """
    Convert a PIL image to a binarised / cleaned PIL image optimised for
    Tesseract digit recognition.

    Steps
    -----
    1. Convert to OpenCV grayscale.
    2. Apply **Otsu** thresholding (more robust than adaptive for
       uniform-background game screenshots).
    3. Light denoising.
    4. Invert if background is predominantly dark so text is black-on-white.
    5. Return as a PIL ``Image`` (mode ``"L"``).

    Parameters
    ----------
    pil_image : PIL.Image.Image
        The input image (RGB or RGBA).

    Returns
    -------
    PIL.Image.Image
        Pre-processed single-channel (grayscale) image ready for Tesseract.
    """
    # RGBA → RGB
    if pil_image.mode == "RGBA":
        pil_image = pil_image.convert("RGB")

    # PIL → OpenCV BGR
    open_cv_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # 1. Grayscale
    gray = cv2.cvtColor(open_cv_bgr, cv2.COLOR_BGR2GRAY)

    # 2. Otsu thresholding → binary (more robust than adaptive for
    #    typical game screenshots with uniform backgrounds)
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    # 3. Light denoise (speckle removal)
    denoised = cv2.fastNlMeansDenoising(binary, h=10)

    # 4. Invert dark backgrounds so text is black-on-white
    if np.mean(denoised) < 127:
        denoised = cv2.bitwise_not(denoised)

    # Return as PIL "L" (grayscale) image
    return Image.fromarray(denoised, mode="L")


# ---------------------------------------------------------------------------
# Score extraction
# ---------------------------------------------------------------------------


def _run_tesseract(
    pil_image: Image.Image,
    psm: int,
) -> str:
    """Run Tesseract on *pil_image* with the given page-segmentation mode
    and return the raw text output."""
    config = CUSTOM_CONFIG.format(psm=psm)
    return pytesseract.image_to_string(pil_image, config=config)


def extract_scores_from_image(
    pil_image: Image.Image,
    min_confidence: float = 0.3,  # kept for API compatibility (unused by Tesseract)
) -> List[float]:
    """
    Extract numeric multiplier scores from an Aviator game screenshot using
    Tesseract OCR.

    Strategy
    --------
    1. Pre-process the image (grayscale + Otsu threshold + denoise).
    2. Prepare a plain grayscale version as a fallback candidate.
    3. For **each** candidate image run Tesseract with **all** PSM modes.
    4. Always also run on the original (un-thresholded) grayscale image which
       often works best for clean screenshots.
    5. Collect candidate-score-sets from every combination; keep the set with
       the **most** valid scores.
    6. Use regex to find decimal numbers (with optional trailing ``x``/``X``).
    7. Filter out obvious false positives (values ≤ 0 or unreasonably high).

    Parameters
    ----------
    pil_image : PIL.Image.Image
        The screenshot / image loaded from clipboard or file.
    min_confidence : float
        **Not used by Tesseract** – kept for API compatibility with the
        original EasyOCR interface.  All Tesseract output is accepted at
        face value (it does not produce a confidence score).

    Returns
    -------
    List[float]
        A list of extracted multiplier scores, sorted in detection order.

    Raises
    ------
    ValueError
        If no valid numeric scores could be extracted from the image, or
        if the Tesseract executable cannot be found.
    """
    # ------------------------------------------------------------------
    # Prepare candidate images
    # ------------------------------------------------------------------
    processed = preprocess_image(pil_image)

    # Plain grayscale (no thresholding) – often best for clean screenshots
    gray_pil = pil_image.convert("L")

    candidates: List[tuple] = [
        (processed, "otsu-threshold"),
        (gray_pil, "plain-grayscale"),
    ]

    # ------------------------------------------------------------------
    # Regex pattern for decimal numbers optionally followed by x/X
    # ------------------------------------------------------------------
    score_pattern = re.compile(r"\b(\d+\.\d+)\s*[xX]?\b")

    # ------------------------------------------------------------------
    # Try every (image, PSM) combination; collect the best score set
    # ------------------------------------------------------------------
    best_scores: List[float] = []

    for img, label in candidates:
        for psm in PSM_MODES:
            text = _run_tesseract(img, psm=psm)
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

            # Parse scores from these lines immediately
            scores: List[float] = []
            seen: set = set()
            for line in lines:
                matches = score_pattern.findall(line)
                for match in matches:
                    clean = match.strip()
                    try:
                        val = round(float(clean), 2)
                    except ValueError:
                        continue
                    # Sanity bounds: Aviator multipliers are typically 1.00–50.00
                    if val <= 0 or val > 200.0:
                        continue
                    key = round(val, 2)
                    if key not in seen:
                        seen.add(key)
                        scores.append(val)

            # Keep the result with the most valid scores
            if len(scores) > len(best_scores):
                best_scores = scores

    # ------------------------------------------------------------------
    # Final check
    # ------------------------------------------------------------------
    if not best_scores:
        raise ValueError(
            "Could not detect any valid scores in the image. "
            "Please ensure the numbers are clearly visible."
        )

    return best_scores


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick self-test with a synthetic image
    import sys

    test_img = Image.new("RGB", (400, 200), color=(30, 30, 30))
    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(test_img)
    try:
        font = ImageFont.truetype("arial.ttf", 32)
    except OSError:
        font = ImageFont.load_default()
    draw.text((20, 30), "1.23", fill=(0, 255, 0), font=font)
    draw.text((20, 80), "5.40x", fill=(255, 255, 0), font=font)
    draw.text((20, 130), "12.30", fill=(255, 255, 255), font=font)

    try:
        extracted = extract_scores_from_image(test_img)
        print(f"Extracted scores: {extracted}")
    except ValueError as e:
        print(f"OCR extraction failed: {e}")
    except RuntimeError as e:
        print(f"Tesseract not installed or not found: {e}")