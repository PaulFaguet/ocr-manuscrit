from pdf2image import convert_from_path
import cv2
import numpy as np
import os

PDF_PATH = "./data/manuscrit.pdf"
PAGES_RANGE = range(20, 21)
DPI = 300


def convert_pdf_to_image(pdf_path: str, page_number: int, dpi: int = 300) -> str:
    """Convertit une page PDF en image PNG."""
    output_path = f"./data/images/page_{page_number:03d}.png"
    pages = convert_from_path(pdf_path, first_page=page_number, last_page=page_number, dpi=dpi)
    pages[0].save(output_path, "PNG")
    return output_path


def segment_page(image_path: str, page_number: int) -> list:
    """Segmente une page en lignes."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    projection = np.sum(binary, axis=1)
    threshold = np.mean(projection) * 0.3

    in_line = False
    lines = []
    start = 0

    for i, val in enumerate(projection):
        if val > threshold and not in_line:
            start = i
            in_line = True
        elif val <= threshold and in_line:
            if i - start > 10:
                lines.append((start, i))
            in_line = False

    # Extraction des lignes
    margin = 15
    line_paths = []
    for i, (y_start, y_end) in enumerate(lines):
        y0 = max(0, y_start - margin)
        y1 = min(img.shape[0], y_end + margin)
        line_img = img[y0:y1, :]
        output_path = f"./data/lines/page_{page_number:03d}_line_{i+1:03d}.png"
        cv2.imwrite(output_path, line_img)
        line_paths.append(output_path)

    return line_paths


if __name__ == "__main__":
    os.makedirs("./data/images", exist_ok=True)
    os.makedirs("./data/lines", exist_ok=True)
    
    total_lines = 0
    for page_num in PAGES_RANGE:
        print(f"\n--- Page {page_num} ---")
        
        image_path = convert_pdf_to_image(PDF_PATH, page_num, DPI)
        print(f"Convertie : {image_path}")
        
        line_paths = segment_page(image_path, page_num)
        print(f"Lignes extraites : {len(line_paths)}")
        total_lines += len(line_paths)

    print(f"\n=== TOTAL : {total_lines} lignes ===")