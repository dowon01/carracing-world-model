import torch
import numpy as np
import matplotlib.pyplot as plt
from world_transformer import WorldTransformer, load_episode_ids
from vq_vae import VQVAE
from config import (FRAMES_PATH, ACTIONS_PATH, EPISODES_PATH, TOKENS_PATH, VQ_VAE_PATH, WORLD_MODEL_PATH,
                    VQ_VAE_CONFIG, WORLD_MODEL_CONFIG, LATENT_SIZE, get_device)
import random

def visualize_imagination():
    device = get_device()
    seq_len = WORLD_MODEL_CONFIG["seq_len"]
    latent_shape = (1, LATENT_SIZE, LATENT_SIZE, VQ_VAE_CONFIG["embedding_dim"])

    # 1. 모델 초기화
    vae = VQVAE(**VQ_VAE_CONFIG).to(device)
    vae.load_state_dict(torch.load(VQ_VAE_PATH, map_location=device))

    world_model = WorldTransformer(**WORLD_MODEL_CONFIG).to(device)
    world_model.load_state_dict(torch.load(WORLD_MODEL_PATH, map_location=device))

    vae.eval()
    world_model.eval()

    # 2. 데이터 준비 (랜덤 시퀀스)
    tokens = np.load(TOKENS_PATH)  # (N, 16, 16)
    actions = np.load(ACTIONS_PATH)  # (N, 3)
    original_frames = np.load(FRAMES_PATH)  # (N, 64, 64, 3)
    episode_ids = load_episode_ids(EPISODES_PATH, len(tokens))

    # 입력 seq_len 프레임과 정답 프레임이 모두 같은 에피소드에 있는 위치만 선택
    while True:
        idx = random.randint(0, len(tokens) - seq_len - 1)
        if episode_ids[idx] == episode_ids[idx + seq_len]:
            break

    input_tokens = torch.LongTensor(tokens[idx:idx+seq_len]).view(1, seq_len, -1).to(device)  # (1, 5, 256)
    # frame[t] -> frame[t+1]을 만드는 행동은 actions[t+1]
    input_actions = torch.FloatTensor(actions[idx+1:idx+seq_len+1]).view(1, seq_len, -1).to(device)  # (1, 5, 3)
    target_token = tokens[idx+seq_len]  # 다음 프레임의 정답 토큰
    last_input_token = tokens[idx+seq_len-1]

    def decode(indices):
        z = vae._vq_vae.get_codebook_entry(indices.view(1, -1), shape=latent_shape)
        return vae._decoder(z).clamp(0, 1).cpu().squeeze(0).permute(1, 2, 0).numpy()

    # 3. 상상하기
    with torch.no_grad():
        logits = world_model(input_tokens, input_actions)  # (1, 5, 256, 512)
        # 마지막 프레임 위치의 출력 = 그 다음 프레임에 대한 예측
        pred_token_indices = torch.argmax(logits[:, -1, :, :], dim=-1)

        # 4. VQ-VAE로 디코딩
        imagined_frame = decode(pred_token_indices)
        target_frame = decode(torch.LongTensor(target_token).to(device))
        last_input_frame = decode(torch.LongTensor(last_input_token).to(device))

        # 원본 프레임 (복원 전)
        raw_frame = original_frames[idx+seq_len] # 디코더를 거치지 않음 (64, 64, 3)

    token_acc = (pred_token_indices.cpu().numpy().reshape(-1) == target_token.reshape(-1)).mean()
    copy_acc = (last_input_token.reshape(-1) == target_token.reshape(-1)).mean()
    print(f"idx={idx}, action={actions[idx+seq_len]}, token acc: model {token_acc:.3f} / copy baseline {copy_acc:.3f}")

    # 5. 시각화
    panels = [
        ("Last Input Frame (VQ recon)", last_input_frame),
        ("Actual Next Frame (VQ recon)", target_frame),
        ("Imagined Next Frame", imagined_frame),
        ("Original Raw Next Frame", raw_frame),
    ]
    plt.figure(figsize=(20, 5))
    for i, (title, img) in enumerate(panels):
        plt.subplot(1, len(panels), i + 1)
        plt.title(title)
        plt.imshow(img)
        plt.axis('off')

    plt.show()

if __name__ == "__main__":
    visualize_imagination()
