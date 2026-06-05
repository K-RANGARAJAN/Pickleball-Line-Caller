import cv2
import torch
import numpy as np
import csv
import argparse
import os
from model import TrackNetV2

def generate_pre_labels():
    parser = argparse.ArgumentParser(description="AI Pre-Label Generator")
    parser.add_argument("--video", type=str, required=True, help="Input video file path")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained .pt weights file")
    parser.add_argument("--output_csv", type=str, required=True, help="Output pre-labeled CSV path")
    args = parser.parse_args()

    if not os.path.exists(args.video) or not os.path.exists(args.weights):
        print("[ERROR] Missing video or weights file."); return

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Launching inference engine on device: {device}")

    # Load baseline model
    model = TrackNetV2(input_channels=9, output_channels=1).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    cap = cv2.VideoCapture(args.video)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # TrackNet standard dimension resize properties
    INPUT_WIDTH, INPUT_HEIGHT = 512, 288

    frames_buffer = []
    pre_labels = {}
    frame_id = 0

    print("[INFO] Processing video frames sequentially...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Preprocess frame image structure to match training inputs
        resized = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
        normalized = resized.astype(np.float32) / 255.0
        transposed = np.transpose(normalized, (2, 0, 1)) # HWC to CHW
        
        frames_buffer.append(transposed)
        
        if len(frames_buffer) < 3:
            # Not enough frames yet for a complete sequence stack
            pre_labels[frame_id] = (0, 0, 0, 0.0)
            frame_id += 1
            continue

        if len(frames_buffer) > 3:
            frames_buffer.pop(0)

        # Stack 3 sequential frames together (9 channels total)
        input_tensor = np.concatenate(frames_buffer, axis=0)
        input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).to(device) # Add batch dimension

        with torch.no_grad():
            output = model(input_tensor)
            heatmap = output.squeeze().cpu().numpy()

        # Find pixel location with the highest confidence value
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(heatmap)

        # Threshold gate: only log it as a guess if the AI is reasonably confident
        if max_val > 0.3:
            # Map the downscaled model pixel prediction back to the true original video resolution
            scale_x = frame_width / INPUT_WIDTH
            scale_y = frame_height / INPUT_HEIGHT
            true_x = int(max_loc[0] * scale_x)
            true_y = int(max_loc[1] * scale_y)
            pre_labels[frame_id] = (true_x, true_y, 1, float(max_val))
        else:
            pre_labels[frame_id] = (0, 0, 0, float(max_val))

        frame_id += 1

    cap.release()

    # Save initial guesses out to a baseline tracking sheet including explicit confidence tracking
    with open(args.output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["frame_id", "ball_x", "ball_y", "is_visible", "confidence"])
        for f_idx in range(frame_id):
            x, y, vis, conf = pre_labels.get(f_idx, (0, 0, 0, 0.0))
            writer.writerow([f_idx, x, y, vis, conf])

    print(f"[SUCCESS] AI baseline pre-label generation completed! Saved to: {args.output_csv}")

if __name__ == "__main__":
    generate_pre_labels()