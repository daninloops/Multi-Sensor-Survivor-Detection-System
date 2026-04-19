import random
import time
import os
from src.paths import LOGS_DIR

last_alert_time = 0
ALERT_COOLDOWN = 10
log_file = os.path.join(LOGS_DIR, "alert_log.txt")

def generate_gps_coordinates():
    latitude = round(random.uniform(12.80, 12.90), 6)
    longitude = round(random.uniform(80.00, 80.10), 6)
    return latitude, longitude

def send_alert(confidence):
    global last_alert_time
    
    current_time = time.time()
    
    if current_time - last_alert_time < ALERT_COOLDOWN:
        return
        
    last_alert_time = current_time
    
    lat, lon = generate_gps_coordinates()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    print("\n================================")
    print("ALERT TRIGGERED\n")
    print("Possible survivor detected!\n")
    print(f"Time: {timestamp}")
    print(f"Location: {lat}, {lon}")
    print(f"Confidence: {confidence:.2f}")
    print("================================\n")
    
    with open(log_file, "a") as f:
        f.write(
            f"{timestamp}, "
            f"{lat}, "
            f"{lon}, "
            f"{confidence}\n"
        )
