#import tensorflow
#import certifi
from keras.applications import MobileNetV3Large # type: ignore
from keras.applications.mobilenet_v3 import preprocess_input, decode_predictions # type: ignore
import cv2  # type: ignore
import numpy as np

MNv3 = MobileNetV3Large(weights='imagenet')

# Initialize webcam | 0 is the defualt number of webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Read and process video stream frame-by-frame
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame. Exiting ...")
        break

    # Resize frame to 224x224 for MobileNetV2
    resized_frame = cv2.resize(frame, (224, 224))

    # Preprocess the image
    x = preprocess_input(np.expand_dims(resized_frame, axis=0))

    # Make predictions
    preds = MNv3.predict(x, verbose=0)
    # Decode the top-1 prediction
    prediction = decode_predictions(preds, top=1)[0][0][1]

    # Overlay the prediction on the frame
    cv2.putText(frame, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Real-time Object Detection', frame)

    # Break the loop with the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()