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

IMAGE_PATH = "assets\\rys_1.2.png"
img_raw = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
(thresh, img_bw) = cv2.threshold(img_raw, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


if img_bw is not None:
    img_resized = cv2.resize(img_bw, (28, 28))
    img_resized = cv2.bitwise_not(img_resized)
    img_normalized = img_resized.astype('float32') / 255.0
    digit_tensor = np.expand_dims(img_normalized, axis=(0, -1))
    print(f"Shape tensora: {digit_tensor.shape}")
    print(f"Wartości pikseli: min={digit_tensor.min():.2f}, max={digit_tensor.max():.2f}")
    img_to_show = np.squeeze(digit_tensor)

    plt.figure(figsize=(4, 4))
    plt.imshow(img_to_show, cmap='gray', vmin=0, vmax=1)
    plt.colorbar(label="Jasność piksela (0=czarny, 1=biały)")
    plt.title("Tak ten obraz widzi sieć")
    plt.show()

    prediction = model.predict(digit_tensor)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    print(f"Predykcja sieci: {predicted_class} (Pewność: {confidence * 100:.2f}%)")
else:
    print(f"Nie znaleziono pliku pod ścieżką: {IMAGE_PATH}")