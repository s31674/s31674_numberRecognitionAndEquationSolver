import os

import cv2
import numpy as np

output_folder = 'assets/database'
os.makedirs(output_folder, exist_ok=True)

def cutting_out_cells(img, table, pic_number):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))

    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    table_grid = cv2.add(horizontal_lines, vertical_lines)
    contours, hierarchy = cv2.findContours(table_grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cells = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 20 and h > 10 and w < img.shape[1] * 0.9 and h < img.shape[0] * 0.9:
            if h >= 100 and h < 400 and w >= 100 and w < 400:
                cells.append((x, y, w, h))
    cells = sorted(cells, key=lambda b: (b[1] // 15, b[0]))

    i = 0
    for j, (x, y, w, h) in enumerate(cells):
        cells_per_table = 32
        pos_in_table = j % cells_per_table
        if pos_in_table < 28:
            margin = 6
            content_x = x + margin
            content_y = y + margin
            content_w = w - 2*margin
            content_h = h - 2*margin
            if content_w <= 0 or content_h <= 0:
                continue
            content_img = img[content_y:content_y + content_h, content_x:content_x + content_w]
            cell_gray = cv2.cvtColor(content_img, cv2.COLOR_BGR2GRAY)
            _, cell_thresh = cv2.threshold(cell_gray, 200, 255, cv2.THRESH_BINARY_INV)
            noise_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            cleaned_thresh = cv2.morphologyEx(cell_thresh, cv2.MORPH_OPEN, noise_kernel, iterations=1)
            final_img = cv2.bitwise_not(cleaned_thresh)
            if cv2.countNonZero(cleaned_thresh) < 40:
                continue

            category = table[i % len(table)]
            folder_path = os.path.join(output_folder, category)
            os.makedirs(folder_path, exist_ok=True)
            filename = os.path.join(folder_path, f"{table[i % len(table)]}_{pic_number}_{i}.png")
            cv2.imwrite(filename, final_img)
            i = i + 1

def main():
    table = [
        "pi", "e", "x", "y", "z", "plus", "minus", "rowna_sie",
        "otwarcie1", "zamkniecie1", "otwarcie2", "zamkniecie2",
        "otwarcie3", "zamkniecie3", "symbol_sumy", "znak_iloczynu",
        "pierwiastek", "dzielenie_slash", "dzielenie_dwukropek",
        "dzielenie_ulamek", "mnozenie_kropka", "mnozenie_gwiazdka",
        "mnozenie_x", "modulo", "calka", "pochodna", "silnia", "bazgrol"
    ]
    for i in range(9, 15):
        full_path = f"assets/tables_to_crop/Zosia_Inz_000{i}.jpg"
        img = cv2.imread(full_path)
        if img is None:
            print(f"Ostrzeżenie: Nie znaleziono pliku '{full_path}'. Pomijam.")
            continue
        print("tne obrazek")
        cutting_out_cells(img, table, i)

if __name__ == "__main__":
    main()