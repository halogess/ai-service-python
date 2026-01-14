"""
PDF image conversion helpers.
"""

import logging
import os
from typing import List, Optional

import fitz

logger = logging.getLogger(__name__)

DEFAULT_DPI = 300
DEFAULT_IMAGE_FORMAT = "jpg"
DEFAULT_ZERO_PAD = 0


def _resolve_output_dir(pdf_path: str, output_dir: Optional[str]) -> str:
    if output_dir:
        return output_dir
    pdf_dir = os.path.dirname(pdf_path)
    if os.path.basename(pdf_dir).lower() == "pdf":
        doc_root = os.path.dirname(pdf_dir)
        return os.path.join(doc_root, "images")
    return os.path.join(pdf_dir, "images")


def convert_pdf_to_images(
    pdf_path: str,
    output_dir: Optional[str] = None,
    dpi: int = DEFAULT_DPI,
    image_format: str = DEFAULT_IMAGE_FORMAT,
    zero_pad: int = DEFAULT_ZERO_PAD,
    overwrite: bool = False,
) -> List[str]:
    """
    Convert PDF pages to images and save them in order.

    Args:
        pdf_path: Full path to PDF.
        output_dir: Output directory (default: document base directory).
        dpi: Render DPI.
        image_format: "png" or "jpg".
        zero_pad: Zero pad page numbers (0 = no padding).
        overwrite: Overwrite existing images if True.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    output_dir = _resolve_output_dir(pdf_path, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    image_format = image_format.lower().lstrip(".")
    if image_format == "jpeg":
        image_format = "jpg"
    if image_format not in ("png", "jpg"):
        raise ValueError(f"Unsupported image format: {image_format}")

    scale = dpi / 72.0
    matrix = fitz.Matrix(scale, scale)

    doc = fitz.open(pdf_path)
    try:
        total_pages = doc.page_count
        if zero_pad is None:
            pad = len(str(total_pages))
        elif zero_pad <= 0:
            pad = 0
        else:
            pad = zero_pad
        image_paths = []

        for page_idx in range(total_pages):
            if pad > 0:
                filename = f"{page_idx + 1:0{pad}d}.{image_format}"
            else:
                filename = f"{page_idx + 1}.{image_format}"
            output_path = os.path.join(output_dir, filename)
            image_paths.append(output_path)

            if not overwrite and os.path.exists(output_path):
                continue

            page = doc[page_idx]
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            pix.save(output_path)

        logger.info("Saved %s page images to: %s", len(image_paths), output_dir)
        return image_paths
    finally:
        doc.close()
