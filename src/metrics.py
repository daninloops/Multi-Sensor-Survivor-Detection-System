import os
from src.paths import LOGS_DIR

class PerformanceMetrics:
    def __init__(self):
        self.total_frames = 0
        self.total_detections = 0
        self.total_alerts = 0
        self.summary_file = os.path.join(LOGS_DIR, "metrics_summary.txt")

    def update_metrics(self, fusion_result):
        self.total_frames += 1
        if fusion_result["survivor_detected"]:
            self.total_detections += 1
            
        if self.total_frames % 10 == 0:
            self.display_metrics()
            self.save_summary()

    def add_alert(self):
        self.total_alerts += 1

    def display_metrics(self):
        detection_rate = (self.total_detections / self.total_frames) * 100 if self.total_frames > 0 else 0
        print("\n--- Runtime Metrics ---")
        print(f"Total Frames: {self.total_frames}")
        print(f"Detections: {self.total_detections}")
        print(f"Alerts: {self.total_alerts}")
        print(f"Detection Rate: {detection_rate:.1f}%")
        print("-----------------------\n")

    def save_summary(self):
        detection_rate = (self.total_detections / self.total_frames) * 100 if self.total_frames > 0 else 0
        with open(self.summary_file, "w") as f:
            f.write(f"Total Frames: {self.total_frames}\n")
            f.write(f"Total Detections: {self.total_detections}\n")
            f.write(f"Total Alerts: {self.total_alerts}\n")
            f.write(f"Detection Rate: {detection_rate:.1f}%\n")
