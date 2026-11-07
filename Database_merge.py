import keras
import numpy as np
import tensorflow as tf

def limit_mnist_class_size(x_mnist, y_mnist, max_samples_per_class):
    x_limited, y_limited = [], []
    np.random.seed(424)
    for cls in range(10):
        indices = np.where(y_mnist == cls)[0]
        selected_indices = np.random.choice(
            indices,
            min(len(indices), max_samples_per_class),
            replace=False
        )
        x_limited.append(x_mnist[selected_indices])
        y_limited.append(y_mnist[selected_indices])

    return np.concatenate(x_limited), np.concatenate(y_limited)

def preprocess_mnist(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, -1)
    image = tf.image.resize(image, [64, 64])
    label = tf.cast(label, tf.int32)
    return image, label

def merging_datasets():
    symbol_train = tf.keras.utils.image_dataset_from_directory(
        directory="assets/fliped_database/",
        labels='inferred',
        label_mode='int',
        color_mode='grayscale',
        batch_size=32,
        image_size=(64, 64),
        validation_split=0.2,
        shuffle=True,
        subset='training',
        seed=123,
    )
    symbol_test = tf.keras.utils.image_dataset_from_directory(
        directory="assets/fliped_database/",
        labels='inferred',
        label_mode='int',
        color_mode='grayscale',
        batch_size=32,
        image_size=(64, 64),
        validation_split=0.2,
        shuffle=True,
        subset='validation',
        seed=123,
    )
    mydb_class_names = symbol_train.class_names
    mnist_classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    all_class_names = mnist_classes + mydb_class_names

    symbol_augmentation = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1.0 / 255.0),
        tf.keras.layers.RandomTranslation(height_factor=0.08, width_factor=0.08, fill_mode="constant", fill_value=0),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(height_factor=(-0.08, 0.08), width_factor=(-0.08, 0.08), fill_mode="constant", fill_value=0)
    ])

    symbol_train = symbol_train.map(lambda x, y: (symbol_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    rescaling = tf.keras.Sequential([tf.keras.layers.Rescaling(1.0 / 255.0)])
    symbol_test = symbol_test.map(lambda x, y: (rescaling(x), y),num_parallel_calls=tf.data.AUTOTUNE)

    symbol_train = symbol_train.map(
        lambda x, y: (x, y + 10), num_parallel_calls=tf.data.AUTOTUNE
    )
    symbol_test = symbol_test.map(
        lambda x, y: (x, y + 10), num_parallel_calls=tf.data.AUTOTUNE
    )

    (x_mnist_train, y_mnist_train), (x_mnist_test, y_mnist_test) = keras.datasets.mnist.load_data()

    x_mnist_train, y_mnist_train = limit_mnist_class_size(x_mnist_train, y_mnist_train, max_samples_per_class=350)

    mnist_train_dataset = tf.data.Dataset.from_tensor_slices(
        (x_mnist_train, y_mnist_train)
    )
    mnist_test_dataset = tf.data.Dataset.from_tensor_slices(
        (x_mnist_test, y_mnist_test)
    )


    mnist_train_dataset = mnist_train_dataset.map(preprocess_mnist, num_parallel_calls=tf.data.AUTOTUNE)
    mnist_test_dataset = mnist_test_dataset.map(preprocess_mnist, num_parallel_calls=tf.data.AUTOTUNE)

    mnist_train_dataset = mnist_train_dataset.shuffle(1000).batch(32)
    mnist_test_dataset = mnist_test_dataset.batch(32)

    train_dataset = mnist_train_dataset.concatenate(symbol_train)
    test_dataset = mnist_test_dataset.concatenate(symbol_test)
    return train_dataset, test_dataset, all_class_names