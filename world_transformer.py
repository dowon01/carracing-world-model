import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import os
from config import MAX_EPISODE_STEPS


def load_episode_ids(episode_path, num_frames):
    # 에피소드 id 파일이 있으면 사용하고, 없으면 CarRacing 최대 길이(1000)로 경계를 추정
    if episode_path is not None and os.path.exists(episode_path):
        return np.load(episode_path)
    return np.arange(num_frames) // MAX_EPISODE_STEPS


# 1. 시퀀스 데이터셋 정의
class WorldSequenceDataset(Dataset):
    """
    data_play.py의 저장 규칙: frames[i]는 actions[i]를 실행한 "결과" 관측값
    -> frame[t]에서 frame[t+1]을 만드는 행동은 actions[t+1]

    입력:  frames[idx : idx+T],     actions[idx+1 : idx+T+1]
    타겟:  frames[idx+1 : idx+T+1]
    """
    def __init__(self, token_path, action_path, seq_len=5, episode_path=None, episodes=None):
        # 토큰과 행동 데이터를 로드
        self.tokens = np.load(token_path) # (N, 16, 16)
        self.actions = np.load(action_path) # (N, 3)
        self.seq_len = seq_len

        # 16x16 토큰을 256개의 시퀀스로 변환
        self.tokens = self.tokens.reshape(len(self.tokens), -1) # (N, 256)
        self.episode_ids = load_episode_ids(episode_path, len(self.tokens))

        # 에피소드 경계를 넘지 않는 시작 인덱스만 사용 (seq_len + 1 프레임이 같은 에피소드여야 함)
        starts = np.arange(len(self.tokens) - seq_len)
        valid = self.episode_ids[starts] == self.episode_ids[starts + seq_len]
        if episodes is not None:
            valid &= np.isin(self.episode_ids[starts], episodes)
        self.starts = starts[valid]

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        idx = self.starts[i]
        # 입력: 현재부터 seq_len만큼의 상태와 "다음 프레임을 만드는" 액션
        states = self.tokens[idx : idx + self.seq_len] # (seq_len, 256)
        actions = self.actions[idx + 1 : idx + self.seq_len + 1] # (seq_len, 3)

        # 타겟: 다음 시점의 상태 (토큰)
        targets = self.tokens[idx + 1 : idx + self.seq_len + 1] # (seq_len, 256)

        return {
            "states": torch.LongTensor(states),
            "actions": torch.FloatTensor(actions),
            "targets": torch.LongTensor(targets)
        }


# 2. 트랜스포머 모델 정의 (next-frame prediction)
class WorldTransformer(nn.Module):
    def __init__(self, num_tokens=512, embedding_dim=128, n_heads=4, n_layers=4, seq_len=5, tokens_per_frame=256, dropout=0.1):
        super().__init__()
        self.num_tokens = num_tokens
        self.seq_len = seq_len
        self.tokens_per_frame = tokens_per_frame

        self.token_embedding = nn.Embedding(num_tokens, embedding_dim)
        self.action_embedding = nn.Linear(3, embedding_dim) # 행동도 임베딩

        # 위치 정보: 프레임 내 위치(256) + 시간 위치(seq_len)로 분리
        self.spatial_embedding = nn.Parameter(torch.randn(1, 1, tokens_per_frame, embedding_dim) * 0.02)
        self.temporal_embedding = nn.Parameter(torch.randn(1, seq_len, 1, embedding_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=n_heads, dim_feedforward=4 * embedding_dim,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, num_tokens) # 다음 프레임 토큰 예측

        # 프레임 단위 causal mask: t번째 프레임의 토큰은 0..t 프레임까지만 볼 수 있음
        frame_idx = torch.arange(seq_len).repeat_interleave(tokens_per_frame)
        self.register_buffer("attn_mask", frame_idx[None, :] > frame_idx[:, None], persistent=False)

    def forward(self, states, actions):
        b, t, s = states.size() # (Batch, seq_len, 256)

        # token 임베딩 (B, T, 256, D)
        token_embeddings = self.token_embedding(states)
        # action 임베딩을 시퀀스 차원에 맞게 확장 (B, T, 1, D)
        action_embeddings = self.action_embedding(actions).unsqueeze(2)

        # action 정보를 각 token에 더함 (conditioning) + 위치 정보 추가
        x = token_embeddings + action_embeddings + self.spatial_embedding + self.temporal_embedding[:, :t]
        x = x.view(b, t * s, -1)

        # 미래 프레임을 보지 못하도록 마스크 적용 후 트랜스포머 연산
        mask = self.attn_mask[: t * s, : t * s]
        x = self.transformer(x, mask=mask)

        # 결과값 출력 (B, T, 256, num_tokens): t번째 위치는 t+1번째 프레임을 예측
        logits = self.output_layer(self.norm(x))
        return logits.view(b, t, s, -1)
