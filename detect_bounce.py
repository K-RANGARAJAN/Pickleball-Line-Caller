import os
import csv
import json
import numpy as np
import cv2

# 1. Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(SCRIPT_DIR, "ball_dataset.csv")
config_path = os.path.join(SCRIPT_DIR, "court_config.json")

if not os.path.exists(csv_path) or not os.path.exists(config_path):
    print("[ERROR] Missing dataset or calibration config file.")
    exit()

# 2. Load Homography Matrix
with open(config_path, 'r') as f:
    config = json.load(f)
src_pts = np.array(config["outer_corners"], dtype=np.float32)
dst_pts = np.array([[0, 0], [400, 0], [400, 880], [0, 880]], dtype=np.float32)
M = cv2.getPerspectiveTransform(src_pts, dst_pts)

# 3. Read dataset and transform to Virtual Space
frames = []
virtual_y_coords = []
pixel_coords = []

with open(csv_path, mode='r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        if row and int(row[3]) == 1:
            f_id = int(row[0])
            bx, by = int(row[1]), int(row[2])
            
            point = np.array([[[bx, by]]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(point, M)
            ty = transformed[0][0][1]
            
            frames.append(f_id)
            pixel_coords.append((bx, by))
            virtual_y_coords.append(ty)

# 4. Smooth the data using a 3-frame moving average to remove clicking shake
smoothed_y = []
for i in range(len(virtual_y_coords)):
    start_idx = max(0, i - 1)
    end_idx = min(len(virtual_y_coords), i + 2)
    smoothed_y.append(np.mean(virtual_y_coords[start_idx:end_idx]))

# 5. Advanced Turning Point Filter with Depth Gate
bounce_detected = False

for i in range(1, len(smoothed_y) - 1):
    incoming_diff = smoothed_y[i] - smoothed_y[i-1]
    outgoing_diff = smoothed_y[i+1] - smoothed_y[i]
    
    # NEW CRITERIA: Turning point must occur deeper than Y=300 to filter out hand-shakiness at the start
    if incoming_diff > 0.1 and outgoing_diff < -0.1 and smoothed_y[i] > 300:
        bounce_detected = True
        detected_frame = frames[i]
        ty = virtual_y_coords[i]
        status = "IN" if 0 <= ty <= 880 else "OUT"
        
        print(f"\n🎯 [BOUNCE CONFIRMED BY MATH ENGINE]")
        print(f"-> Peak Bounce Frame: {detected_frame} (Index {i})")
        print(f"-> Verified Court Depth: Y = {ty:.1f} (Ground Apex)")
        print(f"-> Official Line Call: ** {status} **\n")
        break

if not bounce_detected:
    # Fallback to the absolute physical maximum depth recorded
    max_idx = np.argmax(virtual_y_coords)
    ty = virtual_y_coords[max_idx]
    status = "IN" if 0 <= ty <= 880 else "OUT"
    print(f"🎯 [STATISTICAL MAX APEX DETECTED]")
    print(f"-> Peak Bounce Frame: {frames[max_idx]} (Index {max_idx})")
    print(f"-> Verified Court Depth: Y = {ty:.1f}")
    print(f"-> Official Line Call: ** {status} **\n")