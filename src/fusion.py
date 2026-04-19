def fuse_signals(camera, thermal, sound, gas, movement):
    confidence = 0.0
    
    if camera["detected"]:
        confidence += 0.4
        
    if thermal["human_heat_detected"]:
        confidence += 0.2
        
    if sound["sound_detected"]:
        confidence += 0.15
        
    if gas["co2_level"] == "HIGH":
        confidence += 0.15
        
    if movement["movement_detected"]:
        confidence += 0.1
        
    survivor_detected = confidence > 0.6
    
    return {
        "survivor_detected": survivor_detected,
        "confidence": confidence
    }
