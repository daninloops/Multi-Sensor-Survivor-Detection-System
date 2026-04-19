import time
import os
from src.paths import LOGS_DIR

def log_system_data(camera, thermal, sound, gas, movement, fusion_result):
    log_file = os.path.join(LOGS_DIR, "system_log.txt")
    
    needs_header = not os.path.exists(log_file) or os.path.getsize(log_file) == 0
    
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    with open(log_file, "a") as f:
        if needs_header:
            f.write("timestamp, camera_confidence, temperature, sound, gas, movement, fusion_confidence, survivor_detected\n")
            
        f.write(
            f"{timestamp}, "
            f"{camera['confidence']:.4f}, "
            f"{thermal['temperature']:.2f}, "
            f"{sound['sound_detected']}, "
            f"{gas['co2_level']}, "
            f"{movement['movement_detected']}, "
            f"{fusion_result['confidence']:.4f}, "
            f"{fusion_result['survivor_detected']}\n"
        )
