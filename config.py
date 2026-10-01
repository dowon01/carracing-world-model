import os
import torch

# 코드북 크기 256으로 실험: VQ_NUM_EMBEDDINGS=256 python world_train.py
NUM_EMBEDDINGS = int(os.environ.get("VQ_NUM_EMBEDDINGS") or 512)
EXP_SUFFIX = "" if NUM_EMBEDDINGS == 512 else f"_k{NUM_EMBEDDINGS}"

# 데이터 경로
FRAMES_PATH = "racing_data/play_action_frames.npy"
ACTIONS_PATH = "racing_data/play_actions.npy"
EPISODES_PATH = "racing_data/play_episode_ids.npy"
TOKENS_PATH = f"racing_data/latent_tokens{EXP_SUFFIX}.npy"

# 모델 경로
VQ_VAE_PATH = "model/vq_vae_racing.pth" if NUM_EMBEDDINGS == 512 else f"model/codebook_experiments/vq_vae_k{NUM_EMBEDDINGS}_thr1.0.pth"
WORLD_MODEL_PATH = f"model/world_transformer{EXP_SUFFIX}.pth"

# CarRacing-v3의 최대 에피소드 길이 (에피소드 id 파일이 없을 때 경계 추정용)
MAX_EPISODE_STEPS = 1000

# VQ-VAE 하이퍼파라미터 (모든 스크립트에서 공유)
VQ_VAE_CONFIG = dict(
    num_hiddens=128,
    num_residual_layers=2,
    num_residual_hiddens=32,
    num_embeddings=NUM_EMBEDDINGS,  # 단어장 크기
    embedding_dim=64,
    commitment_cost=0.5,
    dead_code_threshold=1.0,    # 죽은 토큰 교체 기준 (EMA 사용량)
)
LATENT_SIZE = 16    # 64x64 이미지 -> 16x16 토큰

# World Transformer 하이퍼파라미터
WORLD_MODEL_CONFIG = dict(
    num_tokens=VQ_VAE_CONFIG["num_embeddings"],
    embedding_dim=128,
    n_heads=4,
    n_layers=4,
    seq_len=5,
    tokens_per_frame=LATENT_SIZE * LATENT_SIZE,
)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
