import json

from keras import Sequential
import keras
from keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
)
import Database_merge

train_dataset, test_dataset, all_class_names = Database_merge.merging_datasets()

model = Sequential()
model.add(Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(64, 64, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(120, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(38, activation='softmax'))

model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False), optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

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

model.fit(train_dataset, epochs=50, verbose=1, validation_data=test_dataset, callbacks=[es, rlp])
with open("assets/learning_output/class_names.json", "w", encoding="utf-8") as f:
  json.dump(all_class_names, f, ensure_ascii=False, indent=4)
model.save("assets/learning_output/full_model.keras")
print("Model został pomyślnie zapisany!")
