import tensorflow as tf
import cv2
import numpy as np

# Load the model
model = tf.keras.models.load_model('keras_model.h5')

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
video = cv2.VideoCapture(0)

while True:
    check, frame = video.read()
    resize = cv2.resize(frame, (224, 224))
    frame_array = np.asarray(resize)
    normalized_frame_array = (frame_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_frame_array

    prediction = model.predict(data)
    text = "None"  # Default text

    if prediction[0][0] > 0.7:
        text = "Charu"
    elif prediction[0][1] > 0.7:
        text = "Utkarsh"

    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('AI', frame)
    # Stop if the escape key is pressed
    key = cv2.waitKey(1)
    # Stop the program with 'Q' key
    if key == 81 or key == 113:
        break

# Release the VideoCapture object
video.release()
cv2.destroyAllWindows()
