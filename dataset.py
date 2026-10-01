import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 레이싱 게임 이미지 데이터셋 로드 및 전처리
class RacingDataset(Dataset):
    def __init__(self, npy_file):
        # (N, H, W, C) -> (N, C, H, W)로 변경하고 0~1 사이로 정규화
        # (N, 64, 64, 3) -> (N, 3, 64, 64)
        self.data = np.load(npy_file).transpose(0, 3, 1, 2) / 255.0
        self.data = torch.tensor(self.data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
        

if __name__ == "__main__":
    dataset = RacingDataset('racing_data/play_action_frames.npy')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    batch = next(iter(dataloader))
    print(f"배치 데이터 모양: {batch.shape}, 데이터 개수: {len(dataset)}") # 예상 출력: (32, 3, 64, 64)