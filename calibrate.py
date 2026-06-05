import cv2
import json
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Manual Court Calibration Studio")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output", type=str, default="court_config.json", help="Path to save config")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"[ERROR] Could not find video file at: {args.video}")
        exit()

    corners = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(corners) < 4:
                corners.append([x, y])
                print(f"Point {len(corners)} Registered: ({x}, {y})")
                
                # Draw point visual indicator
                cv2.circle(frame_copy, (x, y), 6, (0, 0, 255), -1)
                cv2.putText(frame_copy, str(len(corners)), (x + 10, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("Manual Court Calibration Window", frame_copy)
                
                if len(corners) == 4:
                    print("\nAll 4 corners successfully captured!")
                    print("Order: Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left")
                    print("Press ANY KEY on your keyboard to save and close.")

    print("\nStarting Calibration Engine...")
    print("Instructions: Click the 4 outermost boundaries of the court lines in this exact order:")
    print("1. Top-Left  2. Top-Right  3. Bottom-Right  4. Bottom-Left\n")

    cap = cv2.VideoCapture(args.video)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("[ERROR] Could not extract calibration frame from video stream.")
        exit()

    frame_copy = frame.copy()
    cv2.putText(frame_copy, "Click 4 Court Outer Corners: TL -> TR -> BR -> BL", 
                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Manual Court Calibration Window", frame_copy)
    cv2.setMouseCallback("Manual Court Calibration Window", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(corners) == 4:
        config_data = {"outer_corners": corners}
        with open(args.output, 'w') as f:
            json.dump(config_data, f, indent=4)
        print(f"\n[SUCCESS] Calibration file built flawlessly! Saved to: {args.output}")
    else:
        print("\n[WARNING] Calibration cancelled.")

if __name__ == "__main__":
    main()