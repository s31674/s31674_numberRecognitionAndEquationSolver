import json
import cv2
import numpy as np
import tensorflow as tf

model_learned = tf.keras.models.load_model('assets\\learning_output\\mix_learned.keras')
with open("assets\\learning_output\\sings_classes.json", "r") as f:
    model_classes = json.load(f)

cap = cv2.VideoCapture(0)

while True:
    is_on, frame = cap.read()
    if not is_on:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if w < 35 or h < 35:
            continue

        roi = gray[y:y + h, x:x + w]

        pad = 8
        roi_padded = cv2.copyMakeBorder(
            roi, pad, pad, pad, pad,
            cv2.BORDER_CONSTANT, value=0
        )

        # 5. Dopasowanie do formatu 28x28
        img = cv2.resize(roi_padded, (28, 28))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=(0, -1))  # Kształt: (1, 28, 28, 1)

        # 6. Predykcja właściwym modelem (model_learned, a nie nazwa pliku json!)
        predictions = model_learned.predict(img, verbose=0)
        idx_label = np.argmax(predictions)
        confidence = np.max(predictions)

        print(model_classes[idx_label])

        if confidence > 0.45:
            best_label = model_classes[idx_label]
            label = f"{best_label} ({confidence * 100:.0f}%)"

            color = (0, 255, 0)  # Zielona ramka
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow('Kamerka RPS', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()