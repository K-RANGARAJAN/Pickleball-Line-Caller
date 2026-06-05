import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os

IMG_WIDTH, IMG_HEIGHT = 512, 288
SIGMA = 2.5 

def generate_heatmap(cx, cy, width, height, sigma):
    """Generates a localized 2D Gaussian heatmap layer target around the ball center."""
    heatmap = np.zeros((height, width), dtype=np.float32)
    if cx == 0 and cy == 0:
        return heatmap 
        
    x = np.arange(0, width, 1, float)
    y = np.arange(0, height, 1, float)[:, np.newaxis]
    
    heatmap = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
    return heatmap

class TrackNetDataset(Dataset):
    """Dynamically structures custom video files and CSV coordinate sheets with Data Augmentation."""
    def __init__(self, target_pairs):
        self.samples = []
        
        for video_path, csv_path in target_pairs:
            if not os.path.exists(video_path) or not os.path.exists(csv_path):
                print(f"[WARNING] Skipping pair. File missing: {video_path} or {csv_path}")
                continue
                
            df = pd.read_csv(csv_path)
            cap = cv2.VideoCapture(video_path)
            v_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            v_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                resized = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
                normalized = resized.astype(np.float32) / 255.0
                transposed = np.transpose(normalized, (2, 0, 1)) 
                frames.append(transposed)
            cap.release()
            
            for idx in range(2, len(frames)):
                if idx in df['frame_id'].values:
                    row = df[df['frame_id'] == idx].iloc[0]
                    
                    orig_x, orig_y, visible = row['ball_x'], row['ball_y'], row['is_visible']
                    
                    if visible == 1 and orig_x > 0 and orig_y > 0:
                        model_x = int(orig_x * (IMG_WIDTH / v_width))
                        model_y = int(orig_y * (IMG_HEIGHT / v_height))
                    else:
                        model_x, model_y = 0, 0
                        
                    target_map = generate_heatmap(model_x, model_y, IMG_WIDTH, IMG_HEIGHT, SIGMA)
                    input_stack = np.concatenate([frames[idx-2], frames[idx-1], frames[idx]], axis=0)
                    
                    self.samples.append((input_stack, target_map))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_stack, target_map = self.samples[idx]
        
        # --- DATA AUGMENTATION UPGRADE ---
        # 1. Random Horizontal Flip (Simulates the rally going the opposite direction)
        if np.random.rand() > 0.5:
            input_stack = np.flip(input_stack, axis=2).copy()  # Flip across image width axis
            target_map = np.flip(target_map, axis=1).copy()    # Flip heatmap width axis
            
        # 2. Random Brightness Shift (Simulates fluctuating outdoor court shadows)
        brightness_factor = np.random.uniform(0.8, 1.2)
        input_stack = np.clip(input_stack * brightness_factor, 0.0, 1.0)
        # ----------------------------------

        return torch.tensor(input_stack, dtype=torch.float32), torch.tensor(target_map, dtype=torch.float32).unsqueeze(0)

def train_model():
    from model import TrackNetV2
    
    training_data_pairs = [
        ("Test_recording_2.mp4", "test2_ball_dataset.csv"),
        ("TEST yt video 1.mov", "test_yt_1_ball_dataset.csv")
    ]
    
    print("\n[STEP 1/3] Compiling and packaging tracking layers into memory arrays...")
    dataset = TrackNetDataset(training_data_pairs)
    if len(dataset) == 0:
        print("[ERROR] Dataset compilation arrays returned blank."); return
        
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[STEP 2/3] Hardware deployment initialized on engine: {device}")
    
    model = TrackNetV2(input_channels=9, output_channels=1).to(device)
    
    # --- BUG FIX: Changed from BCEWithLogitsLoss to BCELoss to align with model.py ---
    criterion = nn.BCELoss() 
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    epochs = 15 
    print(f"[STEP 3/3] Training sequence engaged over {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(dataset)
        print(f" -> Epoch {epoch+1:02d}/{epochs:02d} | Net Loss Density: {epoch_loss:.6f}")
        
    output_weight_path = "baseline_tracknet.pt"
    torch.save(model.state_dict(), output_weight_path)
    print(f"\n[SUCCESS] Custom Neural Network weights built natively: {output_weight_path}\n")

if __name__ == "__main__":
    train_model()