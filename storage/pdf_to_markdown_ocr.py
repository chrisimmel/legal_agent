#!/usr/bin/env python3
"""
PDF to Markdown Converter with OCR Support
Extracts text from PDF files (including scanned images) and converts to markdown format.
"""

import fitz  # PyMuPDF
import re
import os
import io
import cv2
import numpy as np
from PIL import Image
import pytesseract
from pathlib import Path
from typing import List, Dict, Optional, Tuple


class PDFToMarkdownConverterOCR:
    """Converts PDF documents to markdown format with OCR support. (Currently unused)"""

    def __init__(self, preserve_formatting: bool = True, enable_ocr: bool = True):
        """
        Initialize the converter.

        Args:
            preserve_formatting: Whether to preserve text formatting
            enable_ocr: Whether to enable OCR for scanned text in images
        """
        self.preserve_formatting = preserve_formatting
        self.enable_ocr = enable_ocr

        # Configure tesseract path if needed (common on macOS)
        if os.path.exists("/opt/homebrew/bin/tesseract"):
            pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"
        elif os.path.exists("/usr/local/bin/tesseract"):
            pytesseract.pytesseract.tesseract_cmd = "/usr/local/bin/tesseract"

    def extract_images_from_page(self, page) -> List[Tuple[np.ndarray, Dict]]:
        """
        Extract images from a PDF page.

        Args:
            page: PyMuPDF page object

        Returns:
            List of tuples containing (image_array, image_info)
        """
        images = []

        # Get image list from page
        image_list = page.get_images()

        for img_index, img in enumerate(image_list):
            try:
                # Get image data
                xref = img[0]
                pix = fitz.Pixmap(page.parent, xref)

                # Convert to PIL Image
                if pix.n - pix.alpha < 4:  # GRAY or RGB
                    img_data = pix.tobytes("png")
                    pil_image = Image.open(io.BytesIO(img_data))
                    img_array = np.array(pil_image)
                else:  # CMYK
                    pix1 = fitz.Pixmap(fitz.csRGB, pix)
                    img_data = pix1.tobytes("png")
                    pil_image = Image.open(io.BytesIO(img_data))
                    img_array = np.array(pil_image)
                    pix1 = None

                # Store image info
                image_info = {
                    "index": img_index,
                    "bbox": img[2],  # bounding box
                    "width": img_array.shape[1],
                    "height": img_array.shape[0],
                }

                images.append((img_array, image_info))
                pix = None

            except Exception as e:
                print(f"Warning: Could not extract image {img_index}: {e}")
                continue

        return images

    def perform_ocr_on_image(self, image: np.ndarray) -> str:
        """
        Perform OCR on an image to extract text.

        Args:
            image: Image as numpy array

        Returns:
            Extracted text from the image
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            # Apply preprocessing to improve OCR accuracy
            # Resize if too small
            if gray.shape[0] < 100 or gray.shape[1] < 100:
                scale_factor = max(200 / gray.shape[0], 200 / gray.shape[1])
                new_height = int(gray.shape[0] * scale_factor)
                new_width = int(gray.shape[1] * scale_factor)
                gray = cv2.resize(gray, (new_width, new_height))

            # Apply thresholding to improve text clarity
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Perform OCR
            text = pytesseract.image_to_string(binary, config="--psm 6")

            return text.strip()

        except Exception as e:
            print(f"Warning: OCR failed on image: {e}")
            return ""

    def extract_text_with_ocr(self, pdf_path: str) -> str:
        """
        Extract text from PDF including OCR for scanned images.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Combined text from embedded text and OCR
        """
        doc = fitz.open(pdf_path)
        all_text = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)

            # Extract embedded text
            embedded_text = page.get_text()

            # Extract text from images using OCR
            ocr_text = ""
            if self.enable_ocr:
                images = self.extract_images_from_page(page)
                for img_array, img_info in images:
                    img_text = self.perform_ocr_on_image(img_array)
                    if img_text:
                        ocr_text += f"\n[Image {img_info['index']}]: {img_text}\n"

            # Combine text
            page_text = embedded_text
            if ocr_text:
                page_text += f"\n{ocr_text}\n"

            all_text.append(page_text)

        doc.close()
        return "\n".join(all_text)

    def extract_text_with_formatting(self, pdf_path: str) -> List[Dict]:
        """
        Extract text from PDF with formatting information.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of dictionaries containing page text and formatting info
        """
        doc = fitz.open(pdf_path)
        pages_data = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)

            # Get text blocks with formatting
            blocks = page.get_text("dict")

            page_data = {
                "page_number": page_num + 1,
                "blocks": blocks["blocks"],
                "text": page.get_text(),
            }

            pages_data.append(page_data)

        doc.close()
        return pages_data

    def extract_text_simple(self, pdf_path: str) -> str:
        """
        Extract plain text from PDF.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Extracted text as string
        """
        doc = fitz.open(pdf_path)
        text = ""

        for page in doc:
            text += page.get_text()

        doc.close()
        return text

    def clean_text_for_markdown(self, text: str) -> str:
        """
        Clean and format text for markdown conversion.

        Args:
            text: Raw extracted text

        Returns:
            Cleaned text ready for markdown conversion
        """
        # Remove excessive whitespace
        text = re.sub(r"\n\s*\n\s*\n", "\n\n", text)
        text = re.sub(r" +", " ", text)

        # Clean up line breaks
        text = text.strip()

        return text

    def detect_headers(self, text: str) -> str:
        """
        Detect and convert potential headers to markdown format.

        Args:
            text: Text to process

        Returns:
            Text with headers converted to markdown
        """
        lines = text.split("\n")
        processed_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                processed_lines.append("")
                continue

            # Simple header detection based on capitalization and length
            if (
                len(line) < 100
                and line.isupper()
                and not line.endswith(".")
                and not any(char.isdigit() for char in line[:3])
            ):
                # Likely a header - convert to markdown
                processed_lines.append(f"## {line.title()}")
            else:
                processed_lines.append(line)

        return "\n".join(processed_lines)

    def convert_to_markdown(
        self, pdf_path: str, output_path: Optional[str] = None
    ) -> str:
        """
        Convert PDF to markdown format with OCR support.

        Args:
            pdf_path: Path to the PDF file
            output_path: Optional path to save the markdown file

        Returns:
            Markdown content as string
        """
        print(f"Converting {pdf_path} to markdown...")

        # Extract text with OCR if enabled
        if self.enable_ocr:
            print("Using OCR for scanned text extraction...")
            text = self.extract_text_with_ocr(pdf_path)
        else:
            # Use original text extraction
            if self.preserve_formatting:
                # Get formatting data for future enhancement
                _ = self.extract_text_with_formatting(pdf_path)
                text = self.extract_text_simple(pdf_path)
            else:
                text = self.extract_text_simple(pdf_path)

        # Clean and format text
        text = self.clean_text_for_markdown(text)
        text = self.detect_headers(text)

        # Add markdown header
        pdf_name = Path(pdf_path).stem
        markdown_content = f"# {pdf_name}\n\n{text}\n"

        # Save to file if output path provided
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)
            print(f"Markdown saved to: {output_path}")

        return markdown_content

    def batch_convert(self, input_dir: str, output_dir: str) -> List[str]:
        """
        Convert multiple PDF files in a directory.

        Args:
            input_dir: Directory containing PDF files
            output_dir: Directory to save markdown files

        Returns:
            List of output file paths
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        pdf_files = list(input_path.glob("*.pdf"))
        output_files = []

        print(f"Found {len(pdf_files)} PDF files to convert...")

        for pdf_file in pdf_files:
            output_file = output_path / f"{pdf_file.stem}.md"
            self.convert_to_markdown(str(pdf_file), str(output_file))
            output_files.append(str(output_file))

        return output_files


def main():
    """Main function to demonstrate usage."""
    # Example usage with OCR
    converter = PDFToMarkdownConverterOCR(preserve_formatting=True, enable_ocr=True)

    # Convert single file
    pdf_path = "data/fake_case_documents_full.pdf"
    if os.path.exists(pdf_path):
        markdown_content = converter.convert_to_markdown(
            pdf_path, "data/fake_case_documents_full_ocr.md"
        )
        print("Conversion completed!")
        print(f"Extracted {len(markdown_content)} characters")
    else:
        print(f"PDF file not found: {pdf_path}")


if __name__ == "__main__":
    main()
