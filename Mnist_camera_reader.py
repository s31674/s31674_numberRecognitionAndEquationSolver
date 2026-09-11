import json
import cv2
import numpy as np
import tensorflow as tf

model_learned = tf.keras.models.load_model('assets\\learning_output\\mnist_learned.keras')
with open("assets\\learning_output\\mnist_classes.json", "r") as f:
    model_classes = json.load(f)

cap = cv2.VideoCapture(0)

while True:
    is_on, frame = cap.read()
    if not is_on:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.resize(gray_frame, (28, 28))
    gray_frame = cv2.bitwise_not(gray_frame)
    gray_frame = gray_frame.astype("float") / 255.0

    gray_frame = np.expand_dims(gray_frame, axis=0)
    gray_frame = np.expand_dims(gray_frame, axis=-1)

    predictions = np.argmax(model_learned.predict(gray_frame))
    print(predictions)