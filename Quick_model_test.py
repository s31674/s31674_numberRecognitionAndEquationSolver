import json

import cv2
import keras
import numpy as np
from keras.ops import nn
from matplotlib import pyplot as plt

model = keras.models.load_model("assets/learning_output/full_model.keras")
with open("assets/learning_output/class_names.json", "r", encoding="utf-8") as f:
  all_class_names = json.load(f)

IMAGE_PATH = "assets/e.png"
img_raw = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
if img_raw is not None:
    (thresh, img_bw) = cv2.threshold(img_raw, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_resized = cv2.resize(img_bw, (64, 64))
    if np.mean(img_resized) > 127:
        img_resized = cv2.bitwise_not(img_resized)
    img_normalized = img_resized.astype('float32') / 255.0
    digit_tensor = np.expand_dims(img_normalized, axis=(0, -1))
    print(f"Shape tensora: {digit_tensor.shape}")
    print(f"Wartości pikseli: min={digit_tensor.min():.2f}, max={digit_tensor.max():.2f}")

    prediction = model.predict(digit_tensor)
    probabilities = nn.softmax(prediction[0]).numpy()

    top_5_indices = np.argsort(probabilities)[::-1][:5]

    prediction = model.predict(digit_tensor)
    probabilities = nn.softmax(prediction[0]).numpy()
    top_5_indices = np.argsort(probabilities)[::-1][:5]

    print("Surowe indeksy zwrócone przez model:", top_5_indices)
    print("Surowe prawdopodobieństwa dla Top 5:", probabilities[top_5_indices])

    print("\n top 5 wyborów")
    for i, class_idx in enumerate(top_5_indices):
        conf = probabilities[class_idx] * 100
        class_name = all_class_names[class_idx]
        print(
            f"{i + 1}. Nazwa: '{class_name}' (Indeks: {class_idx},"
            f" Pewność: {conf:.2f}%)"
        )

    img_to_show = np.squeeze(digit_tensor)
    plt.figure(figsize=(4, 4))
    plt.imshow(img_to_show, cmap='gray', vmin=0, vmax=1)
    plt.colorbar(label="Jasność piksela (0=czarny, 1=biały)")
    plt.title("Tak ten obraz widzi sieć")
    plt.show()
else:
    print(f"Nie znaleziono pliku pod ścieżką: {IMAGE_PATH}")
