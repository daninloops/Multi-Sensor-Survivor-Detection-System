def fuse_signals(camera, thermal, sound, gas, movement):
    confidence = 0.0
    
    # Base spatial validation logic
    def thermal_alignment(camera, thermal):
        if thermal["human_heat_detected"]:
            return 0.1
        else:
            return -0.05

    if camera["detected"]:
        confidence += 0.4
        # Multi-modal thermal validation boost
        confidence += thermal_alignment(camera, thermal)
        
    if thermal["human_heat_detected"]:
        confidence += 0.2
        
    if sound["sound_detected"]:
        confidence += 0.15
        
    if gas["co2_level"] == "HIGH":
        confidence += 0.15
        
    if movement["movement_detected"]:
        confidence += 0.1

    # Sensor Agreement Boost
    agreement_count = 0
    if camera["detected"]: agreement_count += 1
    if thermal["human_heat_detected"]: agreement_count += 1
    if sound["sound_detected"]: agreement_count += 1
    if movement["movement_detected"]: agreement_count += 1
    
    if agreement_count >= 3:
        confidence += 0.1
        
    survivor_detected = confidence > 0.6
    
    return {
        "survivor_detected": survivor_detected,
        "confidence": min(confidence, 1.0) # Cap at 1.0
    }
