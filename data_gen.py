import json
import random
from datetime import datetime, timedelta

def generate_disaster_data(source_id="sim_disaster_001"):
    """Generates a sample disaster/weather data payload."""
    event_types = ["Hurricane", "Flood", "Wildfire", "Earthquake", "Tornado"]
    locations = ["Coastal Region Alpha", "River Valley Beta", "Forest Area Gamma", "City Delta", "Plains Epsilon"]
    severities = ["Category 3", "Major", "High Risk", "Magnitude 5.5", "EF-3"]
    impacts = [
        "Widespread flooding expected.",
        "Structural damage likely in affected zones.",
        "Rapid fire spread possible due to high winds.",
        "Potential for aftershocks and infrastructure damage.",
        "Significant damage to buildings and trees."
    ]

    event_type = random.choice(event_types)
    severity = random.choice(severities) # Note: Severity might not always match event type perfectly in sim

    data = {
        "source_id": source_id,
        "timestamp": (datetime.now() - timedelta(minutes=random.randint(1, 60))).isoformat(),
        "metadata": {"source_system": "SimulationFramework", "confidence_level": round(random.uniform(0.7, 0.95), 2)},
        "data_type": "disaster", # Explicitly set data_type for the parser
        "data": {
            "event_type": event_type,
            "location": random.choice(locations),
            "severity": severity,
            "predicted_impact": random.choice(impacts),
            "confidence_score": round(random.uniform(0.6, 0.98), 2),
            "raw_inference_data": {
                "model_name": "Predictatron 5000",
                "params": {"wind_speed_knots": random.randint(50, 120), "rainfall_mm": random.randint(50, 300)}
            }
        }
    }
    return data

def generate_debris_data(source_id="sim_debris_001"):
    """Generates a sample space debris data payload."""
    object_ids = [f"SATDEB-{random.randint(1000, 9999)}" for _ in range(5)]
    risk_levels = ["High", "Medium", "Low", "Elevated"]

    trajectory = []
    start_time = datetime.now()
    for i in range(5): # Generate 5 trajectory points
        t = start_time + timedelta(minutes=i*10)
        trajectory.append({
            "timestamp": t.isoformat(),
            "position_km": {"x": round(random.uniform(-1000, 1000), 2), "y": round(random.uniform(6000, 8000), 2), "z": round(random.uniform(-500, 500), 2)},
            "velocity_km_s": {"vx": round(random.uniform(-1, 1), 3), "vy": round(random.uniform(7.0, 8.0), 3), "vz": round(random.uniform(-0.5, 0.5), 3)}
        })

    data = {
        "source_id": source_id,
        "timestamp": (datetime.now() - timedelta(minutes=random.randint(1, 30))).isoformat(),
        "metadata": {"tracking_station": "Station Orion", "pass_number": random.randint(100, 200)},
         "data_type": "debris", # Explicitly set data_type for the parser
        "data": {
            "object_id": random.choice(object_ids),
            "trajectory": trajectory,
            "size_estimate_m": round(random.uniform(0.1, 5.0), 2),
            "collision_risk_assessment": {
                "risk_level": random.choice(risk_levels),
                "closest_approach_km": round(random.uniform(0.5, 50.0), 2),
                "target_asset": f"SATCOM-{random.randint(1, 10)}",
                "probability": round(random.random() * 0.01, 6) # Example probability
            },
            "raw_inference_data": {
                "orbit_elements": {"eccentricity": round(random.random()*0.1, 4), "inclination_deg": round(random.uniform(0, 90), 2)},
                "source_model": "OrbitalTrack v2.1"
            }
        }
    }
    return data

if __name__ == "__main__":
    disaster_payload = generate_disaster_data("sim_disaster_test_1")
    debris_payload = generate_debris_data("sim_debris_test_1")

    # Save to files or print
    with open("sample_disaster_payload.json", "w") as f:
        json.dump(disaster_payload, f, indent=4)
    print("Generated sample_disaster_payload.json")

    with open("sample_debris_payload.json", "w") as f:
        json.dump(debris_payload, f, indent=4)
    print("Generated sample_debris_payload.json")

    # Print for easy copying
    # print("\n--- Disaster Payload ---")
    # print(json.dumps(disaster_payload, indent=4))
    # print("\n--- Debris Payload ---")
    # print(json.dumps(debris_payload, indent=4))