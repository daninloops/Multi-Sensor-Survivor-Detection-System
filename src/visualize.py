import pandas as pd
import matplotlib.pyplot as plt
import os
from src.paths import LOGS_DIR

def plot_metrics():
    log_file = os.path.join(LOGS_DIR, "system_log.txt")
    plot_dir = os.path.join(LOGS_DIR, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    if not os.path.exists(log_file):
        print(f"ERROR: Log file not found at {log_file}")
        return

    try:
        # Load System Logs
        df = pd.read_csv(log_file, skipinitialspace=True)
        
        # 1. Plot Confidence Trend
        plt.figure(figsize=(10, 5))
        plt.plot(df['fusion_confidence'], label='Fusion Confidence', color='blue')
        plt.axhline(y=0.6, color='red', linestyle='--', label='Detection Threshold')
        plt.title('Survivor Detection Confidence Trend')
        plt.xlabel('Cycle Index')
        plt.ylabel('Confidence')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, "confidence_plot.png"))
        plt.close()

        # 2. Plot Temperature Trend
        plt.figure(figsize=(10, 5))
        plt.plot(df['temperature'], label='Thermal Sensor (°C)', color='orange')
        plt.axhline(y=34, color='red', linestyle='--', label='Human Heat Threshold')
        plt.title('Biological Temperature readings Over Time')
        plt.xlabel('Cycle Index')
        plt.ylabel('Temperature (°C)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, "temperature_plot.png"))
        plt.close()

        # 3. Plot Detection Accuracy / Rate
        plt.figure(figsize=(10, 5))
        df['survivor_detected'].astype(int).rolling(window=10).mean().plot(color='green', label='Rolling Detection Rate')
        plt.title('Survivor Detection Consistency (Rolling Average)')
        plt.xlabel('Cycle Index')
        plt.ylabel('Detection Probability')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, "detection_rate.png"))
        plt.close()

        print(f"Visualization complete. Graphs saved to: {plot_dir}")
        
    except Exception as e:
        print(f"ERROR: Visualization failed: {e}")

if __name__ == "__main__":
    plot_metrics()
