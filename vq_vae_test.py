import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from dataset import RacingDataset 
from vq_vae import VQVAE
from config import FRAMES_PATH, VQ_VAE_PATH, VQ_VAE_CONFIG, get_device

def visualize_saved_model(model_path, data_path, num_images=5):
    # 1. 장치 설정 
    device = get_device()
    print(f"사용 장치: {device}")

    # 2. 모델 초기화 (학습 때와 동일한 파라미터)
    model = VQVAE(**VQ_VAE_CONFIG).to(device)

    # 3. 가중치 로드
    # map_location은 다른 장치(CPU 등)에서 저장된 파일을 불러올 때 안전하게 매핑
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"모델 로드 완료: {model_path}")

    # 4. 데이터 로드
    dataset = RacingDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=num_images, shuffle=True)
    
    # 5. 복원 및 시각화
    with torch.no_grad():
        originals = next(iter(dataloader)).to(device)
        _, reconstructions, _ = model(originals)

    # 시각화 루프
    plt.figure(figsize=(15, 6))
    for i in range(num_images):
        # 원본 이미지
        plt.subplot(2, num_images, i + 1)
        plt.imshow(originals[i].cpu().permute(1, 2, 0))
        plt.title("Original")
        plt.axis('off')

        # 복원 이미지
        plt.subplot(2, num_images, i + 1 + num_images)
        # 값 범위를 0~1로 맞춰주기 위해 clip 사용
        recon_img = reconstructions[i].cpu().permute(1, 2, 0).numpy()
        plt.imshow(np.clip(recon_img, 0, 1))
        plt.title("Reconstructed")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_saved_model(VQ_VAE_PATH, FRAMES_PATH)