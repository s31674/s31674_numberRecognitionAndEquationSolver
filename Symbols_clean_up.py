import os
from pathlib import Path

import cv2
import numpy as np

def clean_and_center_image(img_raw):
    if len(img_raw.shape) == 3:
        img_raw = img_raw.squeeze()

    img_uint8 = img_raw.astype(np.uint8)
    img_uint8 = 255 - img_uint8
    _, thresh = cv2.threshold(img_uint8, 30, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh)
    clean_thresh = np.zeros_like(thresh)

    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 10:
            clean_thresh[labels == i] = 255

    y_indices, x_indices = np.where(clean_thresh > 0)
    if len(y_indices) > 0 and len(x_indices) > 0:
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        crop = clean_thresh[y_min:y_max + 1, x_min:x_max + 1]
    else:
        crop = clean_thresh

    h, w = crop.shape
    if h > 0 and w > 0:
        aspect_ratio = w / h
        if aspect_ratio > 2.0:
            min_h_needed = int(w * 0.15)

            if h < min_h_needed:
                diff = min_h_needed - h
                pad_top = diff // 2
                pad_bottom = diff - pad_top
                crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=0)
                h, w = crop.shape

        elif aspect_ratio < 0.5:
            min_w_needed = int(h * 0.15)
            if w < min_w_needed:
                diff = min_w_needed - w
                pad_left = diff // 2
                pad_right = diff - pad_left
                crop = cv2.copyMakeBorder(crop, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
                h, w = crop.shape

    max_dim = max(h, w)
    square = np.zeros((max_dim, max_dim), dtype=np.uint8)

    y_offset = (max_dim - h) // 2
    x_offset = (max_dim - w) // 2
    square[y_offset:y_offset + h, x_offset:x_offset + w] = crop

    margin = int(max_dim * 0.40)
    padded = cv2.copyMakeBorder(square, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=0)
    resized = cv2.resize(padded, (64, 64), interpolation=cv2.INTER_AREA)

    return resized[..., None].astype(np.float32)

def main():
    database_path = Path(r"assets/database")
    main_folder = Path(r"assets/fliped_database")
    set_of_symbol_folders = {}
    for pic in database_path.rglob("*"):
        if pic.is_file():
            img = cv2.imread(str(pic), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                folder = pic.parent.name
                if folder not in set_of_symbol_folders:
                    set_of_symbol_folders[folder] = 0
                i = set_of_symbol_folders[folder]
                cleaned_img = clean_and_center_image(img)
                folder_path = main_folder / folder
                folder_path.mkdir(parents=True, exist_ok=True)
                filename = folder_path / f"{folder}_{i}.png"
                cv2.imwrite(str(filename), cleaned_img)
                set_of_symbol_folders[folder] += 1
            else:
                print("Nie udało się wczytać zdjęcia (sprawdź ścieżkę!).")
if __name__ == "__main__":
    main()



