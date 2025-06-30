# anomaly_detection.py
import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Conv3DAutoencoder(nn.Module):
    def __init__(self):
        super(Conv3DAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool3d(2),
            nn.Conv3d(8, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool3d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(16, 8, kernel_size=2, stride=2), nn.ReLU(),
            nn.ConvTranspose3d(8, 1, kernel_size=2, stride=2), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class VideoDataset(Dataset):
    def __init__(self, folder, clip_len=16, resize=(64, 64)):
        self.videos = []
        self.clip_len = clip_len
        self.resize = resize
        self._load_videos(folder)

    def _load_videos(self, folder):
        for file in os.listdir(folder):
            if file.endswith('.mp4'):
                cap = cv2.VideoCapture(os.path.join(folder, file))
                frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret: break
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = cv2.resize(frame, self.resize)
                    frames.append(frame)
                cap.release()
                frames = np.array(frames)
                for i in range(len(frames) - self.clip_len):
                    clip = frames[i:i+self.clip_len]
                    self.videos.append(clip)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        clip = self.videos[idx]
        clip = torch.tensor(clip, dtype=torch.float32).unsqueeze(0) / 255.0
        return clip

def train(model, dataloader, epochs=10):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")
    torch.save(model.state_dict(), "anomaly_model.pth")

def evaluate(model, dataloader):
    model.eval()
    recon_errors = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = batch.to(device)
            output = model(batch)
            loss = nn.functional.mse_loss(output, batch, reduction='none')
            loss = loss.mean(dim=[1,2,3,4])
            recon_errors.extend(loss.cpu().numpy())
    return recon_errors

if __name__ == "__main__":
    train_data = VideoDataset('sample_data/train')
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True)

    test_data = VideoDataset('sample_data/test')
    test_loader = DataLoader(test_data, batch_size=4, shuffle=False)

    model = Conv3DAutoencoder().to(device)

    print("Training model...")
    train(model, train_loader)

    print("Evaluating...")
    errors = evaluate(model, test_loader)
    np.save("results.npy", np.array(errors))
    print("Saved results to results.npy")
