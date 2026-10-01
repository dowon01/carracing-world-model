import torch
import numpy as np
from torch.utils.data import DataLoader
from vq_vae import VQVAE
from dataset import RacingDataset
from config import FRAMES_PATH, TOKENS_PATH, VQ_VAE_PATH, VQ_VAE_CONFIG, LATENT_SIZE, get_device

def generate_latent_dataset(save_path=TOKENS_PATH):
    device = get_device()
    print(f"Using device: {device}")

    # 1. VQ-VAE 모델 로드
    model = VQVAE(**VQ_VAE_CONFIG).to(device)
    model.load_state_dict(torch.load(VQ_VAE_PATH, map_location=device))
    # eval 모드 필수: train 모드면 토큰화 도중 EMA 코드북 업데이트/죽은 토큰 교체가 일어나
    # 저장된 체크포인트의 코드북과 다른 토큰이 생성됨
    model.eval()

    # 2. 데이터셋 로드
    dataset = RacingDataset(FRAMES_PATH)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    latent_tokens = []

    print("image -> latent token 변환 시작...")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch = batch.to(device)

            # encoder -> pre-vq -> vector quantizer -> index 추출
            z = model._encoder(batch)
            z = model._pre_vq_conv(z)
            _, _, _, encoding_indices = model._vq_vae(z)

            # encoding_indices : (batch * 16 * 16, 1) -> (batch, 16, 16)로 변환
            tokens = encoding_indices.view(-1, LATENT_SIZE, LATENT_SIZE).cpu().numpy()
            latent_tokens.append(tokens)

            if (i+1) % 50 == 0:
                print(f"진행 상황: {(i+1)} / {len(dataloader)} batches")

    # 3. 결과 저장
    latent_data = np.concatenate(latent_tokens, axis=0)
    np.save(save_path, latent_data)
    print(f"Latent token 데이터 저장 완료: {save_path} {latent_data.shape}")


if __name__ == "__main__":
    generate_latent_dataset()
