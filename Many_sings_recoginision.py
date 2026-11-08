import json
import os

import cv2
import keras
import numpy as np
from keras.ops import nn

model = keras.models.load_model("assets/learning_output/full_model.keras")
with open("assets/learning_output/class_names.json", "r", encoding="utf-8") as f:
  all_class_names = json.load(f)

def many_sings_on_picture(thresh):
    img_h, img_w = thresh.shape[:2]
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # dla jednej linijki tekstu
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
    bounding_boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        # if w > 0.7 * img_w or h > 0.7 * img_h:
        #     continue
        # mozna dostosowac na np. 10,15
        if 50 <= w and 50 <= h  and 50 <= area:
            aspect_ratio = w / float(h)
            if 0.1 < aspect_ratio < 1.2:
                bounding_boxes.append((x, y, w, h))
    return bounding_boxes

def format_input(x, y, w, h, thresh):
    crop_img = thresh[y:y + h, x:x + w]
    border_size = 40
    crop_img = cv2.copyMakeBorder(
        crop_img,
        top=border_size,
        bottom=border_size,
        left=border_size,
        right=border_size,
        borderType=cv2.BORDER_CONSTANT,
        value=0
    )
    h, w = crop_img.shape[:2]
    max_dim = max(h, w)
    square = np.zeros((max_dim, max_dim), dtype=np.uint8)
    y_offset = (max_dim - h) // 2
    x_offset = (max_dim - w) // 2
    square[y_offset: y_offset + h, x_offset: x_offset + w] = crop_img
    resized = cv2.resize(
        square, (64,64), interpolation=cv2.INTER_AREA
    )
    normalized = resized.astype(np.float32) / 255.0
    tensor_input = np.expand_dims(normalized, axis=0)
    tensor_input = np.expand_dims(
        tensor_input, axis=-1
    )
    return tensor_input, resized

def model_predict(tensor_input, index):
    prediction = model.predict(tensor_input)
    probabilities = nn.softmax(prediction[0]).numpy()
    top_5_indices = np.argsort(probabilities)[::-1][:5]

    print(f"\n--- Symbol nr {index + 1} ---")
    print("\n top 5 wyborów")
    for i, class_idx in enumerate(top_5_indices):
        conf = probabilities[class_idx] * 100
        class_name = all_class_names[class_idx]
        print(
            f"{i + 1}. Nazwa: '{class_name}' (Indeks: {class_idx},"
            f" Pewność: {conf:.2f}%)"
        )
    return top_5_indices[0]

def process_frame(frame):
    if frame is None or frame.size == 0:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bounding_boxes = many_sings_on_picture(thresh)
    reconised_signs_list = []
    for index, bounding_box in enumerate(bounding_boxes):
        x, y, w, h = bounding_box
        tensor_input, resized_view = format_input(x, y, w, h, thresh)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, str(index + 1), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        class_index = model_predict(tensor_input, index)
        sign = all_class_names[class_index]
        reconised_signs_list.append(sign)
    final_output = ",".join(reconised_signs_list)
    print(final_output)
    return frame

def process_video(video_path):

    base_name, ext = os.path.splitext(video_path)
    output_path = f"{base_name}_gotowe{ext}"

    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        print(f"Błąd: Nie można otworzyć pliku wideo: {video_path}")
        return

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vid.get(cv2.CAP_PROP_FPS)
    if fps == 0 or not fps:
        fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)

    frames_total = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    print(f"[Wideo] Renderowanie: {video_path} -> {output_path}")

    while True:
        ret, frame = vid.read()
        if not ret or frame is None:
            break
        frame_copy = frame.copy()
        processed_frame = process_frame(frame_copy)

        if processed_frame is not None:
            out.write(processed_frame)
        current_frame += 1

        if current_frame % 20 == 0 and frames_total > 0:
            progress = (current_frame / frames_total) * 100
            print(f"Postęp: {progress:.1f}%")

    vid.release()
    out.release()
    print(f"[Wideo] Zakończono! Zapisano plik: {output_path}")

# def process_photo():
#     img = cv2.imread("assets/Image(6).png")
#     if img is None:
#         print("Nie udało się wczytać obrazu assets/lena.png")
#         return
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#     bounding_boxes = many_sings_on_picture(thresh)
#     img_display = img.copy()
#     reconised_signs_list = []
#     for index, bounding_box in enumerate(bounding_boxes):
#         x, y, w, h = bounding_box
#         tensor_input, resized_view = format_input(x, y, w, h, thresh)
#         cv2.rectangle(img_display, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         cv2.putText(img_display, str(index + 1), (x, y - 5),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
#
#         max_display_dim = 800
#         h_img, w_img = img_display.shape[:2]
#
#         if max(h_img, w_img) > max_display_dim:
#             scale = max_display_dim / float(max(h_img, w_img))
#             new_w = int(w_img * scale)
#             new_h = int(h_img * scale)
#             img_resized_display = cv2.resize(img_display, (new_w, new_h), interpolation=cv2.INTER_AREA)
#         else:
#             img_resized_display = img_display
#         cv2.imshow("Wykryte symbole na obrazku", img_resized_display)
#         resized_large = cv2.resize(resized_view, (200, 200), interpolation=cv2.INTER_NEAREST)
#         cv2.imshow("Co widzi siec (Input modelu)", resized_large)
#         class_index = model_predict(tensor_input, index)
#         sign = all_class_names[class_index]
#         reconised_signs_list.append(sign)
#         cv2.waitKey(0)
#     final_output = ",".join(reconised_signs_list)
#     print(final_output)

if __name__ == "__main__":
    process_video("assets/video251.mp4")
