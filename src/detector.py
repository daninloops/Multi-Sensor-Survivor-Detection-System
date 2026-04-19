import numpy as np
import tensorflow as tf
import cv2
import os

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

if __name__ == "__main__":
    try:
        detector = HumanDetector()
        
        test_image_path = os.path.join(
            "data",
            "test_sample.jpg"
        )
        
        if not os.path.exists(test_image_path):
            print(f"ERROR: Image not found at {test_image_path}")
        else:
            image = cv2.imread(test_image_path)
            
            if image is None:
                print(f"ERROR: Could not decode image at {test_image_path}")
            else:
                result = detector.predict(image)
                print("\nPrediction Result:")
                print(f"Detected: {result['detected']}")
                print(f"Confidence: {result['confidence']:.2f}")
                
    except Exception as e:
        print(f"Testing Error: {e}")
