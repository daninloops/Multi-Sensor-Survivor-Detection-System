import numpy as np
import tensorflow as tf
import cv2
import os
import time

from src.paths import TFLITE_MODEL_PATH

class HumanDetector:
    def __init__(self):
        try:
            if not os.path.exists(TFLITE_MODEL_PATH):
                raise FileNotFoundError("TFLite model not found.")
                
            self.interpreter = tf.lite.Interpreter(
                model_path=TFLITE_MODEL_PATH
            )

            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            print("TFLite model loaded successfully.")
            
            # Print Input Shape
            input_shape = self.input_details[0]['shape']
            print(f"Model Input Shape: {list(input_shape)}")
            
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            raise
        except Exception as e:
            print(f"ERROR: Failed to load TFLite model: {e}")
            raise

    def preprocess_image(self, image):
        try:
            if image is None:
                raise ValueError("Image is None")
            
            # Resize to 224x224
            img_resized = cv2.resize(image, (224, 224))
            
            # Normalize and convert to float32
            img_normalized = img_resized.astype(np.float32) / 255.0
            
            # Expand dimensions to (1, 224, 224, 3)
            processed_image = np.expand_dims(img_normalized, axis=0)
            
            return processed_image
            
        except Exception as e:
            print(f"ERROR: Failed to preprocess image: {e}")
            raise

    def predict(self, image):
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Set input tensor
            self.interpreter.set_tensor(
                self.input_details[0]['index'],
                processed_image
            )
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output
            output_data = self.interpreter.get_tensor(
                self.output_details[0]['index']
            )
            
            # Extract confidence
            confidence = float(output_data[0][0])
            
            # Create binary detection
            detected = confidence > 0.5
            
            return {
                "detected": detected,
                "confidence": confidence
            }
            
        except Exception as e:
            print(f"ERROR: Prediction failed: {e}")
            raise

    def predict_frame(self):
        if not hasattr(self, 'cap') or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
            
        ret, frame = self.cap.read()
        if not ret:
            print("ERROR: Failed to capture frame.")
            return {"detected": False, "confidence": 0.0}
            
        return self.predict(frame)

    def start_webcam_detection(self):
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("ERROR: Could not open webcam.")
            return
            
        print("Webcam started successfully.")
        print("Press 'q' to exit.")
        
        while True:
            start_time = time.time()
            
            ret, frame = cap.read()
            
            if not ret:
                print("ERROR: Failed to capture frame.")
                break
                
            result = self.predict(frame)
            
            detected = result["detected"]
            confidence = result["confidence"]
            
            end_time = time.time()
            fps = 1 / (end_time - start_time) if (end_time - start_time) > 0 else 0.0
            
            text_info = f"Detected: {detected} | Confidence: {round(confidence, 2)}"
            cv2.putText(frame, text_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if detected else (0, 0, 255), 2)
            
            text_fps = f"FPS: {round(fps, 1)}"
            cv2.putText(frame, text_fps, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            cv2.imshow("Human Detection", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam stopped.")

if __name__ == "__main__":
    try:
        detector = HumanDetector()
        detector.start_webcam_detection()
    except Exception as e:
        print(f"Execution Error: {e}")
