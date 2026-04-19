# Multi-Sensor Survivor Detection System

## Project Structure

*   **src/**: Contains the core Python scripts for machine learning, sensor integration, and logic modules like alert handling, sensor data fusion, and research-grade logging.
*   **data/**: The central directory for datasets, separated into `raw/` for downloaded images, `processed/` for structured data, and split folders (`train/`, `val/`, `test/`) for the machine learning models.
*   **models/**: The directory where the compiled machine learning models (e.g., .h5, .tflite) are saved after training.
*   **logs/**: The directory for storing system and application logs.
    *   `system_log.txt`: Performance data for every sensor and fusion result (CSV format).
    *   `alert_log.txt`: History of all triggered alerts with GPS timestamps.

## Features
*   **AI Vision**: Real-time human detection using TFLite optimized for edge devices.
*   **Sensor Fusion**: Multi-signal logic combining Camera, Thermal, Sound, Gas, and Ultrasonic inputs.
*   **Intelligent Alerts**: Automated alert system with simulated GPS coordinates and 10-second cooldown protection.
*   **Research Ready**: Continuous logging of all sensor confidence levels for performance analysis.
