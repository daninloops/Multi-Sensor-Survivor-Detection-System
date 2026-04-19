from src.paths import ensure_directories
from src.detector import HumanDetector
from src.sensors import (
    read_thermal_sensor,
    read_microphone,
    read_gas_sensor,
    read_ultrasonic
)
from src.fusion import fuse_signals
from src.alert import send_alert
from src.logger import log_system_data
import time
import sys

ensure_directories()

if __name__ == "__main__":
    print("Initializing Multi-Sensor Dashboard...")
    detector = HumanDetector()
    
    print("Press Ctrl+C to exit.")
    print("-" * 40)
    
    try:
        while True:
            camera_data = detector.predict_frame()
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
                send_alert(fusion_result["confidence"])
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
            
            print("-" * 40)
            time.sleep(1.0)  # Pause for readability
            
    except KeyboardInterrupt:
        print("\nExiting System...")
        sys.exit(0)
