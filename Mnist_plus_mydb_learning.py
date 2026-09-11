import json

import cv2
import keras
import numpy as np
import tensorflow as tf
from keras import Sequential, Input
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, RandomFlip, RandomRotation, RandomZoom, \
    RandomContrast, RandomBrightness, Rescaling, Resizing
from keras.src.layers import BatchNormalization, GaussianNoise
from matplotlib import pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

path_to_database = 'assets/database/'

train_my_ds = keras.utils.image_dataset_from_directory(
    directory=path_to_database,
    labels='inferred',
    label_mode='int',
    color_mode='grayscale',
    batch_size=32,
    image_size=(64, 64),
    validation_split=0.2,
    subset='training',
    seed=424)

validation_my_ds = keras.utils.image_dataset_from_directory(
    directory=path_to_database,
    labels='inferred',
    label_mode='int',
    color_mode='grayscale',
    batch_size=32,
    image_size=(64, 64),
    validation_split=0.2,
    subset='validation',
    seed=424
)

mydb_class_names = train_my_ds.class_names

def dataset_to_numpy(ds):
    x_list, y_list = [], []
    for images, labels in ds:
        x_list.append(images.numpy())
        y_list.append(labels.numpy())
    return np.vstack(x_list), np.concatenate(y_list)

x_train_local, y_train_local = dataset_to_numpy(train_my_ds)
x_val_local, y_val_local = dataset_to_numpy(validation_my_ds)

x_train_local = 255.0 - x_train_local
x_val_local = 255.0 - x_val_local

augmentation = Sequential([
        Input(shape=(64, 64, 1)),
        # RandomFlip("horizontal"),
        RandomRotation(0.03),
        RandomZoom(0.2, fill_mode="nearest"),
        RandomContrast(0.4),
        RandomBrightness(factor=(-0.3,0.3)),
        GaussianNoise(stddev=0.05)
    ])
def augment_data(x_data, y_data, factor=30):
    aug_x, aug_y = [x_data], [y_data]
    for _ in range(factor - 1):
        augmented_images = augmentation(x_data, training=True).numpy()
        aug_x.append(augmented_images)
        aug_y.append(y_data)
    return np.vstack(aug_x), np.concatenate(aug_y)
x_train_local_aug, y_train_local_aug = augment_data(x_train_local, y_train_local, factor=10)

(x_mnist_train, y_mnist_train), (x_mnist_val, y_mnist_val) = keras.datasets.mnist.load_data()
mnist_classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

def limit_mnist_class_size(x_mnist, y_mnist, max_samples_per_class=3000):
    x_limited, y_limited = [], []
    for cls in range(10):
        indices = np.where(y_mnist == cls)[0]
        selected_indices = np.random.choice(
            indices,
            min(len(indices), max_samples_per_class),
            replace=False
        )
        x_limited.append(x_mnist[selected_indices])
        y_limited.append(y_mnist[selected_indices])

    return np.vstack(x_limited), np.concatenate(y_limited)
x_mnist_train, y_mnist_train = limit_mnist_class_size(x_mnist_train, y_mnist_train, max_samples_per_class=3000)

x_mnist_train = x_mnist_train[..., None].astype("float32")
x_mnist_val = x_mnist_val[..., None].astype("float32")

x_mnist_train = tf.image.resize(x_mnist_train, (64,64), method='bicubic').numpy()
x_mnist_val = tf.image.resize(x_mnist_val, (64,64), method='bicubic').numpy()

all_class_names = mnist_classes + mydb_class_names

y_train_local_aug = y_train_local_aug + len(mnist_classes)
y_val_local = y_val_local + len(mnist_classes)

x_train = np.concatenate((x_mnist_train, x_train_local_aug), axis=0)
y_train_int = np.concatenate((y_mnist_train, y_train_local_aug), axis=0)

x_val = np.concatenate((x_mnist_val, x_val_local), axis=0)
y_val_int = np.concatenate((y_mnist_val, y_val_local), axis=0)

x_train = x_train / 255.0
x_val = x_val / 255.0

y_train = keras.utils.to_categorical(y_train_int, num_classes=38)
y_val = keras.utils.to_categorical(y_val_int, num_classes=38)

train_indices = np.arange(x_train.shape[0])
np.random.seed(424)
np.random.shuffle(train_indices)
x_train, y_train = x_train[train_indices], y_train[train_indices]

val_indices = np.arange(x_val.shape[0])
np.random.shuffle(val_indices)
x_val, y_val = x_val[val_indices], y_val[val_indices]

model = Sequential([
    Input(shape=(64, 64, 1)),
    Conv2D(32, (3, 3), activation='relu', padding="same"),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu', padding="same"),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu', padding="same"),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Flatten(),
    # Dense(128, activation='relu'),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(38, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

es = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=8,
    min_delta=1e-3,
    mode='min'
)
rlp =keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_delta=1e-3,
    min_lr=1e-5,
    mode='min',
)

class_weights_vals = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_int),
    y=y_train_int
)

class_weight_dict = dict(enumerate(class_weights_vals))

model.fit(
    x_train, y_train,
    batch_size=100,
    epochs=50,
    validation_data=(x_val, y_val),
    # class_weight=class_weight_dict,
    callbacks=[es, rlp]
)
# with open("assets/learning_output/sings_classes.json", "w") as f:
#     json.dump(all_class_names, f)
# model.save("assets/learning_output/mix_learned.keras")

def prepare_digit_roi(img_raw):
    _, thresh = cv2.threshold(img_raw, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    h, w = thresh.shape
    if h > w:
        pad = (h - w) // 2
        square = cv2.copyMakeBorder(thresh, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=0)
    else:
        pad = (w - h) // 2
        square = cv2.copyMakeBorder(thresh, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=0)

    margin = int(square.shape[0] * 0.15)
    padded = cv2.copyMakeBorder(square, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=0)

    resized = cv2.resize(padded, (64, 64), interpolation=cv2.INTER_AREA)
    normalized = resized.astype('float32') / 255.0
    return normalized.reshape(1, 64, 64, 1)

IMAGE_PATH = "assets\\rys_1.1.png"
img_raw = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
(thresh, img_bw) = cv2.threshold(img_raw, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


if img_bw is not None:
    digit_tensor = prepare_digit_roi(img_raw)

    print(f"Shape tensora: {digit_tensor.shape}")
    print(f"Wartości pikseli: min={digit_tensor.min():.2f}, max={digit_tensor.max():.2f}")

    img_to_show = digit_tensor.reshape(64,64)  # Redukcja do wymiaru (28, 28)

    plt.figure(figsize=(4, 4))
    plt.imshow(img_to_show, cmap='gray', vmin=0, vmax=1)
    plt.colorbar(label="Jasność piksela (0=czarny, 1=biały)")
    plt.title("Tak ten obraz widzi sieć")
    plt.show()

    prediction = model.predict(digit_tensor)
    predicted_index = np.argmax(prediction)

    predicted_label = all_class_names[predicted_index]

    print(f"Rozpoznany symbol: {predicted_label}")
    confidence = np.max(prediction)
    print(f"Predykcja sieci: {predicted_label} (Pewność: {confidence * 100:.2f}%)")
else:
    print(f"Nie znaleziono pliku pod ścieżką: {IMAGE_PATH}")