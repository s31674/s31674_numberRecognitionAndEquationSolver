import json
import matplotlib.pyplot as plt
import keras
from keras import Sequential, Input
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
import numpy as np
import cv2
from keras.src.legacy.preprocessing.image import ImageDataGenerator

(x_train, y_train), (x_val, y_val) = keras.datasets.mnist.load_data()
mnist_classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

x_train = x_train.astype('float32') / 255.0
x_val = x_val.astype('float32') / 255.0

x_train = np.expand_dims(x_train, axis=-1)
x_val = np.expand_dims(x_val, axis=-1)

y_train = keras.utils.to_categorical(y_train, 10)
y_val = keras.utils.to_categorical(y_val, 10)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(84, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, min_delta=1e-3, mode='min')
rlp = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3,  min_lr=1e-6,
                                        mode='min')

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
datagen.fit(x_train)

model.fit(
    x_train, y_train,
    batch_size=100,
    epochs=30,
    validation_data=(x_val, y_val),
    callbacks=[es, rlp]
)

# with open("assets/learning_output/mnist_classes.json", "w") as f:
#     json.dump(mnist_classes, f)
# model.save("assets/learning_output/mnist_learned.keras")


# # 2. FUNKCJA INSPEKCJI I PREDYKCJI WYCIĘTEJ CYFRY
# def inspect_and_predict(crop_bgr, model):
#     """
#     Przetwarza wycięty kontur cyfry, wyświetla jego podgląd Matplotlibem
#     i wykonuje predykcję modelem Keras.
#     """
#     # A. Konwersja do odcieni szarości
#     gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
#
#     # B. Binarizacja (odwrócenie kolorów: cyfra biała na czarnym tle)
#     _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
#
#     # C. Skalowanie do wymiaru 28x28 pikseli
#     resized = cv2.resize(thresh, (28, 28), interpolation=cv2.INTER_AREA)
#
#     # D. Normalizacja zakresu pikseli do [0.0, 1.0]
#     normalized = resized.astype('float32') / 255.0
#
#     # E. Tworzenie tensora (1, 28, 28, 1) dla Keras
#     digit_tensor = np.expand_dims(normalized, axis=(0, -1))
#
#     # F. CONTROL CHECK W KONSOLI
#     print(f"\n--- INSPEKCJA WYCINKA ---")
#     print(f"Shape tensora: {digit_tensor.shape}")
#     print(f"Zakres pikseli: min={digit_tensor.min():.2f}, max={digit_tensor.max():.2f}")
#
#     # G. WIZUALIZACJA MATPLOTLIB (Opcja 1)
#     img_to_show = np.squeeze(digit_tensor)
#     plt.figure(figsize=(6, 6))
#     plt.imshow(img_to_show, cmap='gray', vmin=0, vmax=1)
#     plt.colorbar(label="Jasność")
#     plt.title("Wycinek widziany przez sieć")
#     plt.show()
#
#     # H. PREDYKCJA
#     prediction = model.predict(digit_tensor, verbose=0)
#     predicted_class = np.argmax(prediction)
#     confidence = np.max(prediction)
#     print(f"Predykcja: {predicted_class} ({confidence * 100:.1f}%)")
#
#     return predicted_class, confidence
#
#
# # 3. PRZETWARZANIE WIDEO / OBRAZÓW
# def video():
#     vid = cv2.VideoCapture("assets/test_video.mp4")
#
#     width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = vid.get(cv2.CAP_PROP_FPS)
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#
#     out = cv2.VideoWriter(
#         "assets/video_done.mp4",
#         fourcc,
#         fps,
#         (width, height),
#         isColor=True
#     )
#
#     frames_total = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
#     current_frame = 0
#
#     print("Renderowanie filmu")
#
#     while True:
#         ret, frame = vid.read()
#         if not ret:
#             break
#
#         processed_frame = frame.copy()
#
#         # Przykład znalezienia konturów obiektów (wycięcie fragmentu obrazu):
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         _, thresh_frame = cv2.threshold(gray_frame, 127, 255, cv2.THRESH_BINARY_INV)
#         contours, _ = cv2.findContours(thresh_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#         for cnt in contours:
#             x, y, w, h = cv2.boundingRect(cnt)
#
#             # Filtrowanie zbyt małych wykryć
#             if w > 50 and h > 50:
#                 # Wycięcie obszaru z cyfrą (ROI)
#                 digit_crop = frame[y:y + h, x:x + w]
#
#                 # Wywołanie inspekcji na pierwszej klatce lub określonym warunku
#                 if current_frame == 0:
#                     inspect_and_predict(digit_crop, model)
#
#                 # Narysowanie ramki na klatce wyjściowej
#                 cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
#         out.write(processed_frame)
#         current_frame += 1
#
#         if current_frame % 10 == 0:
#             print(f"Postęp: {current_frame / frames_total:.2f}")
#
#     vid.release()
#     out.release()
#     print("Done")
#
#
# if __name__ == "__main__":
#     video()



IMAGE_PATH = "assets\\rys_1.2.png"
img_raw = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
(thresh, img_bw) = cv2.threshold(img_raw, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


if img_bw is not None:
    img_resized = cv2.resize(img_bw, (28, 28))

    # Odwrócenie kolorów jeśli zdjęcie to czarny rysunek na białym tle:
    img_resized = cv2.bitwise_not(img_resized)

    # Normalizacja [0, 1]
    img_normalized = img_resized.astype('float32') / 255.0

    # Tworzenie tensora dla Keras: wymiary (1, 28, 28, 1)
    digit_tensor = np.expand_dims(img_normalized, axis=(0, -1))

    # C. SPRAWDZENIE W KONSOLI
    print(f"Shape tensora: {digit_tensor.shape}")
    print(f"Wartości pikseli: min={digit_tensor.min():.2f}, max={digit_tensor.max():.2f}")

    # D. WIZUALIZACJA Z MATPLOTLIB
    img_to_show = np.squeeze(digit_tensor)  # Redukcja do wymiaru (28, 28)

    plt.figure(figsize=(4, 4))
    plt.imshow(img_to_show, cmap='gray', vmin=0, vmax=1)
    plt.colorbar(label="Jasność piksela (0=czarny, 1=biały)")
    plt.title("Tak ten obraz widzi sieć")
    plt.show()

    # E. TESTOWE WYKONANIE PREDYKCJI
    prediction = model.predict(digit_tensor)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    print(f"Predykcja sieci: {predicted_class} (Pewność: {confidence * 100:.2f}%)")
else:
    print(f"Nie znaleziono pliku pod ścieżką: {IMAGE_PATH}")