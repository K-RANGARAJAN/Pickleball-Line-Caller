import cv2
import os
import csv
import argparse

parser = argparse.ArgumentParser(description="AI-Assisted Verification Tracking Studio")
parser.add_argument("--video", type=str, required=True, help="Path to input video file")
parser.add_argument("--csv", type=str, required=True, help="Path to save/load tracking dataset")
args = parser.parse_args()

video_path = args.video
csv_path = args.csv

# Load existing entries if the pre-label generator already ran
dataset_map = {}
if os.path.exists(csv_path):
    print(f"[INFO] Found pre-label blueprint. Loading coordinates directly...")
    with open(csv_path, mode='r') as f:
        reader = csv.reader(f)
        next(reader) # skip header
        for row in reader:
            if row:
                f_idx = int(row[0])
                bx = int(row[1])
                by = int(row[2])
                vis = int(row[3])
                # Backwards compatible safety fallback if the old 4-column layout is loaded
                conf = float(row[4]) if len(row) > 4 else (1.0 if vis == 1 else 0.0)
                dataset_map[f_idx] = [bx, by, vis, conf]

cap = cv2.VideoCapture(video_path)
frame_id = 0
current_frame = None
click_registered_this_frame = False
autopilot = False # Tracks whether Auto-Pilot Mode is toggled active

def click_event(event, x, y, flags, param):
    """Overwrites the AI's guess with a precise manual correction click."""
    global frame_id, current_frame, click_registered_this_frame, dataset_map
    if event == cv2.EVENT_LBUTTONDOWN:
        dataset_map[frame_id] = [x, y, 1, 1.0] # Overwrite coordinate state with human input (100% confidence)
        print(f"Correction Logged Frame {frame_id:04d} -> X: {x}, Y: {y}")
        click_registered_this_frame = True
        
        img_copy = current_frame.copy()
        cv2.circle(img_copy, (x, y), 6, (0, 255, 0), -1) # Green for human correction
        cv2.imshow("Labeling Studio", img_copy)

cv2.namedWindow("Labeling Studio", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("Labeling Studio", click_event)

print("\n=======================================================")
print("     MODEL-IN-THE-LOOP TRACKNET VERIFICATION STUDIO    ")
print("=======================================================")
print("-> Magenta Circle = AI's Pre-Label Guess")
print("1. If AI guess is CORRECT: Simply press 'n' to accept and advance.")
print("2. If AI guess is WRONG: Click the true center to overwrite it.")
print("3. Press 's' to manually mark frame as invisible.")
print("4. Press 'b' to undo/rewind frame context.")
print("5. Press 'a' to toggle AUTO-PILOT (AI auto-approves values >= 75% confident)")
print("6. Press 'q' to save progress and exit.")
print("=======================================================\n")

while cap.isOpened():
    # --- AUTO-PILOT CRITERIA GATE ---
    # Fast-forwards past frames where the AI locked onto a ball with high confidence (>= 0.75)
    if autopilot and frame_id in dataset_map and dataset_map[frame_id][2] == 1 and dataset_map[frame_id][3] >= 0.75:
        frame_id += 1
        continue

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()
    if not ret: break
    
    current_frame = frame.copy()
    click_registered_this_frame = False
    
    # If the AI made a guess and marked it visible, draw a prompt cue on screen
    if frame_id in dataset_map and dataset_map[frame_id][2] == 1:
        ax, ay = dataset_map[frame_id][0], dataset_map[frame_id][1]
        c_score = dataset_map[frame_id][3]
        cv2.circle(current_frame, (ax, ay), 8, (255, 0, 255), 2) # Magenta outline target
        cv2.putText(current_frame, f"AI Guess ({c_score:.2f}) [Press 'n' to accept]", (ax + 12, ay - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    while True:
        cv2.imshow("Labeling Studio", current_frame)
        
        # Speed trick: if you manual-clicked, advance instantly!
        if click_registered_this_frame:
            frame_id += 1
            break
            
        key = cv2.waitKey(1) & 0xFF
        
        # Accept current prediction state (AI or fallback) and advance
        if key == ord('n'):
            if frame_id not in dataset_map:
                dataset_map[frame_id] = [0, 0, 0, 0.0]
            frame_id += 1
            break

        # Toggle Auto-Pilot configuration
        elif key == ord('a'):
            autopilot = not autopilot
            print(f"[STUDIO SYSTEM] Auto-Pilot Active Status: {autopilot}")
            # Break inner frame loop to re-evaluate the auto-pilot skip statement instantly
            break
            
        # Hard skip/invisible override
        elif key == ord('s'):
            dataset_map[frame_id] = [0, 0, 0, 0.0]
            print(f"Logged Frame {frame_id:04d} -> Set Invisible")
            frame_id += 1
            break
            
        # Rewind handling
        elif key == ord('b'):
            if frame_id > 0:
                frame_id -= 1
                break
                
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            
            # Flush completely updated tracking array out to the file safely
            with open(csv_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["frame_id", "ball_x", "ball_y", "is_visible", "confidence"])
                for f_k in sorted(dataset_map.keys()):
                    writer.writerow([f_k, dataset_map[f_k][0], dataset_map[f_k][1], dataset_map[f_k][2], dataset_map[f_k][3]])
            print(f"\n[SAVED] Progress recorded to dataset: {csv_path}")
            exit()

cap.release()
cv2.destroyAllWindows()

# Final write-out step upon completing video lifecycle loop
with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["frame_id", "ball_x", "ball_y", "is_visible", "confidence"])
    for f_k in sorted(dataset_map.keys()):
        writer.writerow([f_k, dataset_map[f_k][0], dataset_map[f_k][1], dataset_map[f_k][2], dataset_map[f_k][3]])
print("\nDataset generation finalized.")