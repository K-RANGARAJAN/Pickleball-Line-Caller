import cv2
import os
import json
import csv
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description="Clean Segmented Pickleball Line Caller UI")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--csv", type=str, required=True, help="Path to short segmented tracking CSV")
    parser.add_argument("--config", type=str, default="court_config.json", help="Path to court config map")
    args = parser.parse_args()

    if not os.path.exists(args.config) or not os.path.exists(args.csv) or not os.path.exists(args.video):
        print("[ERROR] Missing required pipeline files. Make sure path metrics are correct."); return

    # Load Auto-Calibration Config
    with open(args.config, 'r') as f:
        config = json.load(f)
    src_pts = np.array(config["outer_corners"], dtype=np.float32)
    dst_pts = np.array([[0, 0], [400, 0], [400, 880], [0, 880]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Re-format court corners into an integer NumPy array for standard drawing operations
    draw_pts = src_pts.astype(np.int32).reshape((-1, 1, 2))

    print("Pre-computing isolated bounce sequences...")
    all_data = []
    with open(args.csv, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            if row and int(row[3]) == 1:
                f_id = int(row[0])
                bx, by = int(row[1]), int(row[2])
                
                point = np.array([[[bx, by]]], dtype=np.float32)
                transformed = cv2.perspectiveTransform(point, M)
                all_data.append({
                    "frame_id": f_id,
                    "pixel": (bx, by),
                    "virtual_x": transformed[0][0][0],
                    "virtual_y": transformed[0][0][1]
                })

    # Group into continuous windows
    segments = []
    if all_data:
        current_segment = [all_data[0]]
        for i in range(1, len(all_data)):
            if all_data[i]["frame_id"] - all_data[i-1]["frame_id"] > 5:
                segments.append(current_segment)
                current_segment = [all_data[i]]
            else:
                current_segment.append(all_data[i])
        segments.append(current_segment)

    detected_bounces = {}
    frames = [d["frame_id"] for d in all_data]
    frame_to_data = {d["frame_id"]: d for d in all_data}

    # Find the exact bounce peak inside each isolated window
    for idx, seg in enumerate(segments):
        if len(seg) < 3: continue
        seg_y = [d["virtual_y"] for d in seg]
        smoothed_y = [np.mean(seg_y[max(0, k-1):min(len(seg_y), k+2)]) for k in range(len(seg_y))]
        peak_index = np.argmax(smoothed_y)
        
        if peak_index == 0 or peak_index == len(smoothed_y) - 1:
            peak_index = len(seg) // 2
            
        b_frame = seg[peak_index]["frame_id"]
        v_x = seg[peak_index]["virtual_x"]
        v_y = seg[peak_index]["virtual_y"]
        
        is_inside = (0 <= v_x <= 400) and (0 <= v_y <= 880)
        detected_bounces[b_frame] = "IN" if is_inside else "OUT"
        print(f"[ENGINE ENFORCED] Bounce Found -> Frame {b_frame} | Position: ({v_x:.1f}, {v_y:.1f}) -> {detected_bounces[b_frame]}")

    # Visual Engine Loop
    cap = cv2.VideoCapture(args.video)
    frame_id = 0
    trail_history = [] # Stores rolling pixel trail coordinates

    print("\n--- BROADCAST ENGINE ACTIVE ---")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_id += 1

        # --- DRAW THE COURT BOUNDARY OVERLAY ---
        # Renders the auto-calibrated court layout perimeter on every video frame state
        cv2.polylines(frame, [draw_pts], isClosed=True, color=(0, 255, 255), thickness=3)

        # Managing the rolling yellow trail history
        if frame_id in frames:
            trail_history.append(frame_to_data[frame_id]["pixel"])
            if len(trail_history) > 12: # Keep trailing line compact
                trail_history.pop(0)
        else:
            trail_history.clear() # Clear trail instantly when out of bounce window

        # Draw the trailing flight line if inside a window
        if trail_history:
            for i in range(1, len(trail_history)):
                cv2.line(frame, trail_history[i-1], trail_history[i], (0, 255, 255), 3)
            # Draw current tracking lead dot
            cv2.circle(frame, trail_history[-1], 6, (0, 255, 255), -1)

        # Trigger Bounce Event Visuals
        if frame_id in detected_bounces:
            call_status = detected_bounces[frame_id]
            bx, by = frame_to_data[frame_id]["pixel"]
            color = (0, 255, 0) if call_status == "IN" else (0, 0, 255)

            # Precision crosshair target overlay
            cv2.line(frame, (bx - 40, by), (bx + 40, by), color, 4)
            cv2.line(frame, (bx, by - 40), (bx, by + 40), color, 4)
            cv2.circle(frame, (bx, by), 14, (255, 255, 255), -1)
            cv2.circle(frame, (bx, by), 6, color, -1)

            # High-end broadcast lower third banner graphic
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, frame.shape[0] - 140), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

            text = f"BOUNCE CALL: {call_status}!"
            cv2.putText(frame, text, (60, frame.shape[0] - 50), cv2.FONT_HERSHEY_DUPLEX, 2.2, color, 5)

            cv2.imshow("Automated Referee Simulator Engine", frame)
            print(f"-> Freeze Frame Event {frame_id}: Ball is {call_status}!")
            if cv2.waitKey(2500) & 0xFF == ord('q'): break # Freeze display for review
        else:
            cv2.imshow("Automated Referee Simulator Engine", frame)
            if cv2.waitKey(25) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()