import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

path = "assets/"
database_path = "database/"

# naprawić cięcie, bo na razie tylko do 5 dobrze tnie
def crop_image(image, table, i_from_pic):
    size = 235
    gap_row = 70
    gap_col = 5
    for i in range(1,3):
        x_start, y_start = 300, 350
        if i == 2:
           y_start = 1620
        for x in range(len(table)):
            x_end = x_start + size
            y_end = y_start + size
            image_copy = image.copy()
            ready_pic = image_copy[y_start:y_end, x_start:x_end]
            save_path = path + database_path + table[x]
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            file_name = f"crop_{x}_{i_from_pic}_{i}.png"
            cv.imwrite(os.path.join(save_path, file_name), ready_pic)
            if (x + 1) % 8 == 0:
                y_start = y_end + gap_row
                x_start = 300
            else:
                x_start = x_end + gap_col

def main():
    table = ["pi", "e", "x", "y", "z", "plus", "minus", "rowna_sie", "otwarcie1", "zamkniecie1", "otwarcie2", "zamkniecie2",
             "otwarcie3", "zamkniecie3", "symbol_sumy", "znak_iloczynu", "pierwiastek", "dzielenie_slash", "dzielenie_dwukropek",
             "dzielenie_ulamek", "mnozenie_kropka", "mnozenie_gwiazdka", "mnozenie_x", "modulo", "calka", "pochodna", "silnia", "bazgrol"]
    for i in range(10, 15):
        filename = f"tables_to_crop/Zosia_Inz_000{i}.jpg"
        full_path = os.path.join(path, filename)
        original_img = cv.imread(full_path)
        if original_img is None:
            print(f"Nie znaleziono pliku: {full_path}")
            continue
        img_copy = original_img.copy()
        crop_image(img_copy, table,i)


if __name__ == "__main__":
    main()
