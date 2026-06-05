import cv2
import os
import csv
import argparse

# Setup command-line parsing to handle any clip dynamically
parser = argparse.ArgumentParser(description="High-Speed TrackNet Data Labeling Studio with Auto-Advance & Undo")
parser.add_argument("--video", type=str, required=True, help="Path to input video file")
parser.add_argument("--csv", type=str, required=True, help="Path to save tracking dataset")
args = parser.parse_args()

video_path = args.video
csv_path = args.csv

cap = cv2.VideoCapture(video_path)

# Verify video accessibility
if not cap.isOpened():
    print(f"\n[ERROR] OpenCV could not open video file at:\n{video_path}")
    print("Please double check that the filename matches exactly and isn't corrupted.\n")
    exit()

# Initialize tracking schema header if it doesn't exist yet
if not os.path.exists(csv_path):
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["frame_id", "ball_x", "ball_y", "is_visible"])

# Maintain dynamic tracking of true frame context
frame_id = 0
current_frame = None
click_registered_this_frame = False

def click_event(event, x, y, flags, param):
    """Logs target coordinates directly locked to the active video frame structure."""
    global frame_id, current_frame, click_registered_this_frame
    if event == cv2.EVENT_LBUTTONDOWN and not click_registered_this_frame:
        with open(csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([frame_id, x, y, 1]) # 1 = visible
        
        print(f"Logged Frame {frame_id:04d} -> X: {x}, Y: {y}")
        click_registered_this_frame = True
        
        # Draw dynamic green confirmation circle over clicked pixel
        img_copy = current_frame.copy()
        cv2.circle(img_copy, (x, y), 6, (0, 255, 0), -1)
        cv2.imshow("Labeling Studio", img_copy)

cv2.namedWindow("Labeling Studio", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("Labeling Studio", click_event)

print("\n=======================================================")
print("   TRACKNET LIGHTNING LABELING STUDIO (AUTO + UNDO)    ")
print("=======================================================")
print("1. Click once directly on the center of the ball.")
print("   -> Window will AUTOMATICALLY advance to the next frame!")
print("2. If the ball is hidden/obscured, press 's' to skip.")
print("3. PRESS 'b' ON KEYBOARD TO UNDO THE PREVIOUS FRAME(S).")
print("4. Press 'q' to safely save dataset progress and quit.")
print("=======================================================\n")

# Process sequential video states
while cap.isOpened():
    # Force the video reader position explicitly to our tracked frame_id index
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()
    if not ret:
        print("Reached the end of the video file.")
        break
    
    current_frame = frame
    click_registered_this_frame = False # Reset confirmation flag for current frame
    
    # Trim data rows ahead of our current position if we rewound into past states
    if os.path.exists(csv_path):
        with open(csv_path, mode='r') as file:
            rows = list(csv.reader(file))
        if len(rows) > 1:
            header = rows[0]
            # Retain only frames that strictly exist BEFORE our active index pointer
            valid_rows = [r for r in rows[1:] if int(r[0]) < frame_id]
            with open(csv_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header)
                writer.writerows(valid_rows)
    
    while True:
        cv2.imshow("Labeling Studio", current_frame)
        
        # SPEED TRICK: If mouse click was registered, break inner loop immediately to load next frame
        if click_registered_this_frame:
            frame_id += 1
            break
            
        key = cv2.waitKey(1) & 0xFF
        
        # Log frame explicitly as an invisible point (Skip hotkey)
        if key == ord('s'):
            with open(csv_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([frame_id, 0, 0, 0]) # 0 = invisible
            print(f"Logged Frame {frame_id:04d} -> Ball Mark Skipped (Invisible)")
            frame_id += 1
            break
            
        # THE BACK/ERASE REWIND MECHANISM
        elif key == ord('b'):
            if frame_id > 0:
                frame_id -= 1  # Drop index pointer back by one frame unit step
                print(f"[UNDO] Rewinding to Frame {frame_id:04d}. Bad data cleared.")
                break
            else:
                print("[WARNING] Already sitting at the initial frame. Cannot rewind further.")
            
        # Clean terminal resource handling termination
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            print(f"\nStudio closed successfully. Progress saved to: {csv_path}")
            exit()

cap.release()
cv2.destroyAllWindows()
print("\nDataset generation finalized.")