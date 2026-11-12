import os
import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class DigitRecognizer:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def preprocess_image(self, image):
        # Resize to 28x28 pixels
        resized_image = cv2.resize(image, (28, 28))
        # Convert to grayscale
        if resized_image.ndim == 3:
            gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = resized_image

        if gray_image.mean() > 127:
            gray_image = 255 - gray_image
        # Normalize pixel values
        normalized_image = gray_image / 255.0
        # Reshape for model input
        reshaped_image = normalized_image.reshape(1, 28, 28, 1)
        return reshaped_image

    def predict_digit(self, image):
        preprocessed_image = self.preprocess_image(image)
        predictions = self.model.predict(preprocessed_image, verbose=0)
        predicted_digit = np.argmax(predictions)
        return predicted_digit
    
    
class ImageProcessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Image at path {image_path} could not be loaded.")

    def convert_to_grayscale(self):
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def apply_gaussian_blur(self, kernel_size=(5, 5)):
        return cv2.GaussianBlur(self.image, kernel_size, 0)

    def detect_edges(self, low_threshold=100, high_threshold=200):
        gray_image = self.convert_to_grayscale()
        return cv2.Canny(gray_image, low_threshold, high_threshold)

    def save_image(self, output_path, image):
        cv2.imwrite(output_path, image)


def select_image_file():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        return filedialog.askopenfilename(
            title="Select a digit image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*"),
            ],
        )
    finally:
        root.destroy()
        
def main():
    model_path = os.path.join(BASE_DIR, "mnist_model.keras")
    # โมเดลหลักเก็บอยู่ที่ models/ (fallback ให้สคริปต์เวอร์ชันเดิมยังใช้งานได้)
    if not os.path.isfile(model_path):
        model_path = os.path.join(BASE_DIR, "models", "mnist_model.keras")
    recognizer = None
    if os.path.isfile(model_path):
        recognizer = DigitRecognizer(model_path)
        print(f"Loaded digit recognition model: {model_path}")
    else:
        print(f"Digit recognition model not found: {model_path}")

    print("Opening image file picker...")
    image_path = select_image_file()

    if image_path:
        print(f"Selected file: {image_path}")
        try:
            processor = ImageProcessor(image_path)

            gray_image = processor.convert_to_grayscale()
            blurred_image = processor.apply_gaussian_blur()
            edges = processor.detect_edges()

            processor.save_image(os.path.join(BASE_DIR, "gray_image.jpg"), gray_image)
            processor.save_image(os.path.join(BASE_DIR, "blurred_image.jpg"), blurred_image)
            processor.save_image(os.path.join(BASE_DIR, "edges.jpg"), edges)
            if recognizer is not None:
                predicted_digit = recognizer.predict_digit(processor.image)
                print(f"Predicted digit: {predicted_digit}")
            print("Image processing completed successfully.")
        except ValueError as error:
            print(f"Skipping image processing: {error}")
    else:
        print("No image selected; skipping image processing.")

    capture_video()

def capture_video():
    cap = None
    camera_index = None

    for index in range(3):
        candidate = cv2.VideoCapture(index)
        if candidate.isOpened():
            cap = candidate
            camera_index = index
            break
        candidate.release()

    if cap is None:
        print("Error: Could not open any available camera (checked indexes 0-2).")
        return

    try:
        print(f"Camera {camera_index} opened. Press 'q' in the video window to exit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            cv2.imshow('Frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()