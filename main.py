from src.paths import ensure_directories
from src.yolo_detector import YOLODetector
from src.sensors import (
    read_thermal_sensor,
    read_microphone,
    read_gas_sensor,
    read_ultrasonic
)
from src.fusion import fuse_signals
from src.alert import send_alert
from src.logger import log_system_data
from src.metrics import PerformanceMetrics
import time
import sys
import json
import os

ensure_directories()

if __name__ == "__main__":
    # Load Deployment Config
    config_path = os.path.join("deploy", "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {"fps_limit": 10}

    print("Initializing Multi-Sensor Dashboard (YOLO Mode)...")
    detector = YOLODetector()
    metrics = PerformanceMetrics()
    
    print("Press Ctrl+C to exit.")
    print("-" * 40)
    
    try:
        while True:
            # Capturing frame for YOLO
            import cv2
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Failed to capture frame.")
                cap.release()
                continue
            
            camera_data = detector.detect_objects(frame)
            thermal_data = read_thermal_sensor()
            sound_data = read_microphone()
            gas_data = read_gas_sensor()
            movement_data = read_ultrasonic()
            
            fusion_result = fuse_signals(
                camera_data,
                thermal_data,
                sound_data,
                gas_data,
                movement_data
            )
            
            # CLI Dashboard output
            print(f"Camera: {camera_data['detected']} ({camera_data['confidence']:.2f})")
            print(f"Thermal: {thermal_data['temperature']:.1f}°C")
            print(f"Sound: {'YES' if sound_data['sound_detected'] else 'NO'}")
            print(f"Gas: {gas_data['co2_level']}")
            print(f"Movement: {'YES' if movement_data['movement_detected'] else 'NO'}")
            print(f"\nFinal Confidence: {fusion_result['confidence']:.2f}")
            
            if fusion_result["survivor_detected"]:
                print("STATUS: SURVIVOR DETECTED")
                if send_alert(fusion_result["confidence"]):
                    metrics.add_alert()
            else:
                print("STATUS: NO SURVIVOR")
                
            print("\nLogging system data...")
            log_system_data(
                camera_data,
                thermal_data,
                sound_data,
                gas_data,
                movement_data,
                fusion_result
            )
            
            metrics.update_metrics(fusion_result)
            
            # Show the frame with YOLO boxes
            cv2.imshow("Multi-Sensor Survivor Detection (YOLO)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            print("-" * 40)
            
            # Deployment FPS limiting
            time.sleep(1 / config.get("fps_limit", 10))
            
            cap.release()
            
    except KeyboardInterrupt:
        print("\nExiting System...")
        cv2.destroyAllWindows()
        sys.exit(0)
    finally:
        cv2.destroyAllWindows()
