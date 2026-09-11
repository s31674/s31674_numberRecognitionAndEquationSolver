import json

import keras
from keras import Sequential, Input
from keras.src.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, RandomFlip, RandomRotation, RandomZoom, \
    RandomContrast, RandomBrightness, Rescaling

path_to_database = 'assets/database/'

train_ds = keras.utils.image_dataset_from_directory(
    directory=path_to_database,
    labels='inferred',
    label_mode='categorical',
    color_mode='grayscale',
    batch_size=32,
    image_size=(28, 28),
    validation_split=0.2,
    subset='training',
    seed=424)

validation_ds = keras.utils.image_dataset_from_directory(
    directory=path_to_database,
    labels='inferred',
    label_mode='categorical',
    color_mode='grayscale',
    batch_size=32,
    image_size=(28, 28),
    validation_split=0.2,
    subset='validation',
    seed=424
)

class_names = train_ds.mydb_class_names

augmentation = Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(32, (3, 3), activation='relu'),
        RandomFlip("horizontal"),
        RandomRotation(0.2),
        RandomZoom(0.2),
        RandomContrast(0.2),
        RandomBrightness(0.2)
    ])

model = Sequential()
Input(shape=(28, 28, 1)),
model.add(Rescaling(1. / 255)),
model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(strides=2))
model.add(Conv2D(filters=48, kernel_size=(5,5), padding='valid', activation='relu'))
model.add(MaxPooling2D(strides=2))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(28, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

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

model.fit(train_ds, epochs=50, validation_data=validation_ds, callbacks=[es, rlp])
with open("assets/learning_output/sings_classes.json", "w") as f:
    json.dump(train_ds.mydb_class_names, f)
model.save("assets/learning_output/signs_learned.keras")