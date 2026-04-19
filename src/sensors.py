import random

def read_thermal_sensor():
    temperature = random.uniform(20.0, 38.0)
    return {
        "temperature": temperature,
        "human_heat_detected": temperature > 34
    }

def read_microphone():
    return {
        "sound_detected": random.choice([True, False])
    }

def read_gas_sensor():
    level = random.choice(["LOW", "MEDIUM", "HIGH"])
    return {
        "co2_level": level
    }

def read_ultrasonic():
    return {
        "movement_detected": random.choice([True, False])
    }
