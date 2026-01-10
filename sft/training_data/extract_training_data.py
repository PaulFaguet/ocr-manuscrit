from pdf2image import convert_from_path
import cv2
import numpy as np
import os

PDF_PATH = "./data/pdf/manuscrit.pdf"
PAGES_RANGE = range(31, 32)
DPI = 300


def convert_pdf_to_image(pdf_path: str, page_number: int, dpi: int = 300) -> str:
    output_path = f"./data/images/page_{page_number:03d}.png"
    pages = convert_from_path(pdf_path, first_page=page_number, last_page=page_number, dpi=dpi)
    pages[0].save(output_path, "PNG")
    return output_path


def adjust_line_height(binary, y0, y1):
    h, w = binary.shape
    min_density = 0.008 * w 

    # vers le haut
    for y in range(y0 - 1, max(y0 - 20, 0), -1):
        if np.sum(binary[y] > 0) > min_density:
            y0 = y
        else:
            break

    # vers le bas (plus large : jambages)
    for y in range(y1, min(y1 + 40, h)):
        if np.sum(binary[y] > 0) > min_density:
            y1 = y + 1
        else:
            break

    return y0, y1


def segment_page(image_path: str, page_number: int):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []

    h_img, w_img = img.shape

    # Inverser les images (noir <-> blanc)
    binary = cv2.adaptiveThreshold(
        img,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=35,
        C=15
    )

    # Nettoyage
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # Projection horizontale
    projection = np.sum(binary > 0, axis=1)

    # Seuil dynamique pour la hauteur des lignes
    threshold = 0.05 * np.max(projection)

    raw_lines = []
    in_line = False
    start_y = 0

    for y, value in enumerate(projection):
        if value > threshold and not in_line:
            in_line = True
            start_y = y
        elif value <= threshold and in_line:
            in_line = False
            raw_lines.append((start_y, y))

    if in_line:
        raw_lines.append((start_y, h_img))

    img_debug = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    line_paths = []

    # Filtrage, marges et sauvegarde
    for i, (y0, y1) in enumerate(raw_lines):
        height = y1 - y0
        # if height < 15:
            # continue

        # margin_y = 5
        # y0 = max(0, y0 - margin_y)
        # y1 = min(h_img, y1 + margin_y)

        # Ajustement dynamique de la hauteur
        y0, y1 = adjust_line_height(binary, y0, y1)

        # Marge de sécurité légère
        # margin_y = 3
        y0 = max(0, y0 - 2)
        y1 = min(h_img, y1 + 4)

        # Crop ligne
        line_img = img[y0:y1, 0:w_img]

        output_path = f"./data/lines/page_{page_number:03d}_line_{i+1:03d}.png"
        cv2.imwrite(output_path, line_img)
        line_paths.append(output_path)

        # Debug visuel
        cv2.rectangle(
            img_debug,
            (0, y0),
            (w_img, y1),
            (0, 0, 255),
            2
        )

    # Image debug
    # cv2.imwrite(f"./data/debug_{page_number:03d}.png", img_debug)

    return line_paths


if __name__ == "__main__":
    os.makedirs("./data/images", exist_ok=True)
    os.makedirs("./data/lines", exist_ok=True)
    
    total_lines = 0
    for page_num in PAGES_RANGE:
        print(f"\nPage {page_num}")
        image_path = convert_pdf_to_image(PDF_PATH, page_num, DPI)
        print(f"Convertie : {image_path}")
        
        line_paths = segment_page(image_path, page_num)
        print(f"Lignes extraites : {len(line_paths)}")
        total_lines += len(line_paths)
    
    print(f"\nTotal : {total_lines} lignes")