import cv2
import json
import argparse
import os
import requests
import base64
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Smart Court Auto-Calibration")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--output", type=str, default="court_config.json", help="Path to save config")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"[ERROR] Missing video path: {args.video}"); return

    API_KEY = "B1IriYIrUZCXD0uq4elR"
    MODEL_ENDPOINT = "pickleball-court-line-dvjn8/1"
    API_URL = f"https://detect.roboflow.com/{MODEL_ENDPOINT}?api_key={API_KEY}"

    cap = cv2.VideoCapture(args.video)
    ret, frame = cap.read()
    cap.release()

    if not ret: return

    print("[INFO] Requesting AI Polygon map...")
    _, buffer = cv2.imencode('.jpg', frame)
    img_str = base64.b64encode(buffer).decode('utf-8')

    response = requests.post(API_URL, data=img_str, headers={"Content-Type": "application/x-www-form-urlencoded"})
    data = response.json()
    predictions = data.get("predictions", [])
    
    if not predictions or not predictions[0].get("points"):
        print("[FAIL] No polygon detected.")
        return

    pts = np.array([[p['x'], p['y']] for p in predictions[0]["points"]])

    # Get the 4 corners of whatever the AI found (could be full court, could be half court)
    s = pts.sum(axis=1)
    diff = pts[:, 0] - pts[:, 1]
    
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmax(diff)]
    bl = pts[np.argmin(diff)]

    detected_corners = [
        [int(tl[0]), int(tl[1])], [int(tr[0]), int(tr[1])],
        [int(br[0]), int(br[1])], [int(bl[0]), int(bl[1])]
    ]

    # --- SMART HOMOGRAPHY EXTRAPOLATION ---
    # We will measure the ratio of the top edge to the bottom edge.
    # If the top edge is too wide, it means the AI stopped at the net instead of the back line.
    top_width = np.linalg.norm(np.array(tl) - np.array(tr))
    bottom_width = np.linalg.norm(np.array(bl) - np.array(br))
    
    if top_width / bottom_width > 0.6: 
        print("[INFO] AI only detected the front half of the court. Extrapolating back lines mathematically...")
        
        # Real-world coordinates of a Pickleball front-half (20ft wide, 22ft deep)
        # Let's map it: Top-Left (Net), Top-Right (Net), Bottom-Right (Baseline), Bottom-Left (Baseline)
        src_real = np.array([[0, 22], [20, 22], [20, 44], [0, 44]], dtype=np.float32)
        dst_pixels = np.array([tl, tr, br, bl], dtype=np.float32)
        
        # Calculate the 3D perspective matrix
        matrix = cv2.getPerspectiveTransform(src_real, dst_pixels)
        
        # Where SHOULD the true back corners (0,0 and 20,0) exist on the screen?
        true_back_real = np.array([[[0, 0], [20, 0]]], dtype=np.float32)
        true_back_pixels = cv2.perspectiveTransform(true_back_real, matrix)
        
        # Overwrite the Top-Left and Top-Right with our math-generated corners
        detected_corners[0] = [int(true_back_pixels[0][0][0]), int(true_back_pixels[0][0][1])]
        detected_corners[1] = [int(true_back_pixels[0][1][0]), int(true_back_pixels[0][1][1])]
        print("[SUCCESS] Missing backline successfully calculated via perspective transform!")
    else:
        print("[SUCCESS] AI successfully detected the full court natively.")

    # Save finalized configuration map
    config_data = {"outer_corners": detected_corners}
    with open(args.output, 'w') as f:
        json.dump(config_data, f, indent=4)

    # Draw the boundary
    pts_to_draw = np.array(detected_corners, np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts_to_draw], isClosed=True, color=(0, 255, 255), thickness=4)

    cv2.imshow("Automated Homography Extrapolation", cv2.resize(frame, (854, 480)))
    cv2.waitKey(4000)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()