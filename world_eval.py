import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from vq_vae import VQVAE
from world_transformer import WorldTransformer, load_episode_ids
from world_train import split_episodes
from config import (FRAMES_PATH, ACTIONS_PATH, EPISODES_PATH, TOKENS_PATH, VQ_VAE_PATH, WORLD_MODEL_PATH,
                    VQ_VAE_CONFIG, WORLD_MODEL_CONFIG, LATENT_SIZE, EXP_SUFFIX, get_device)

# 비교용 고정 행동 (steer, gas, brake)
FIXED_ACTIONS = {
    "noop": [0.0, 0.0, 0.0],
    "left": [-1.0, 0.0, 0.0],
    "right": [1.0, 0.0, 0.0],
    "gas": [0.0, 0.5, 0.0],
}


@torch.no_grad()
def rollout(model, context, action_seq, horizon):
    """
    예측한 프레임을 다시 입력으로 넣어 horizon 스텝 앞까지 예측 (autoregressive)
    context: (B, T, 256) 실제 토큰 T 프레임
    action_seq: (B, T-1+horizon, 3) = 각 프레임 "다음에" 취한 행동 (context의 두 번째 프레임을 만든 행동부터)
    return: (B, horizon, 256)
    """
    seq_len = context.size(1)
    states = context
    preds = []
    for k in range(horizon):
        logits = model(states, action_seq[:, k:k + seq_len])
        next_tokens = logits[:, -1].argmax(-1)
        preds.append(next_tokens)
        states = torch.cat([states[:, 1:], next_tokens[:, None]], dim=1)
    return torch.stack(preds, dim=1)


@torch.no_grad()
def decode(vae, tokens):
    # tokens: (..., 256) -> 이미지 (..., 3, 64, 64)
    shape = tokens.shape[:-1]
    flat = tokens.reshape(-1, LATENT_SIZE * LATENT_SIZE)
    z = vae._vq_vae.get_codebook_entry(flat, (flat.size(0), LATENT_SIZE, LATENT_SIZE, VQ_VAE_CONFIG["embedding_dim"]))
    return vae._decoder(z).clamp(0, 1).view(*shape, 3, 64, 64)


def evaluate(horizon=15, stride=5, batch_size=16, out_prefix=f"model/rollout{EXP_SUFFIX}"):
    device = get_device()
    seq_len = WORLD_MODEL_CONFIG["seq_len"]

    vae = VQVAE(**VQ_VAE_CONFIG).to(device)
    vae.load_state_dict(torch.load(VQ_VAE_PATH, map_location=device))
    model = WorldTransformer(**WORLD_MODEL_CONFIG).to(device)
    model.load_state_dict(torch.load(WORLD_MODEL_PATH, map_location=device))
    vae.eval()
    model.eval()

    tokens = np.load(TOKENS_PATH).reshape(-1, LATENT_SIZE * LATENT_SIZE)
    actions = np.load(ACTIONS_PATH).astype(np.float32)
    frames = torch.tensor(np.load(FRAMES_PATH).transpose(0, 3, 1, 2) / 255.0, dtype=torch.float32)
    episode_ids = load_episode_ids(EPISODES_PATH, len(tokens))

    _, valid_episodes = split_episodes()
    span = seq_len - 1 + horizon
    starts = np.array([i for i in range(0, len(tokens) - span, stride)
                       if episode_ids[i] in valid_episodes and episode_ids[i] == episode_ids[i + span]])
    print(f"검증 에피소드 {valid_episodes.tolist()}, 롤아웃 시작점 {len(starts)}개, horizon {horizon}")

    example_pos = len(starts) // 2


    names = ["model", "copy", "recon_floor"]
    mse = {n: [] for n in names}
    acc = {n: [] for n in ["model", "copy"]}
    action_diff = {f"{a}_vs_{b}": [] for a, b in [("left", "right"), ("left", "noop"), ("right", "noop"), ("gas", "noop")]}

    for b in range(0, len(starts), batch_size):
        s = starts[b:b + batch_size]
        context = torch.tensor(np.stack([tokens[i:i + seq_len] for i in s]), device=device)
        action_seq = torch.tensor(np.stack([actions[i + 1:i + seq_len + horizon] for i in s]), device=device)
        true_tokens = torch.tensor(np.stack([tokens[i + seq_len:i + seq_len + horizon] for i in s]), device=device)
        true_frames = torch.stack([frames[i + seq_len:i + seq_len + horizon] for i in s]).to(device)
        last_frame = torch.stack([frames[i + seq_len - 1] for i in s]).to(device)

        pred_tokens = rollout(model, context, action_seq, horizon)
        pred_frames = decode(vae, pred_tokens)

        mse["model"].append(((pred_frames - true_frames) ** 2).mean((2, 3, 4)).cpu())
        mse["copy"].append(((last_frame[:, None] - true_frames) ** 2).mean((2, 3, 4)).cpu())
        mse["recon_floor"].append(((decode(vae, true_tokens) - true_frames) ** 2).mean((2, 3, 4)).cpu())
        acc["model"].append((pred_tokens == true_tokens).float().mean(-1).cpu())
        acc["copy"].append((context[:, -1:] == true_tokens).float().mean(-1).cpu())

        fixed_frames = {}
        for name, a in FIXED_ACTIONS.items():
            seq = action_seq.clone()
            seq[:, seq_len - 1:] = torch.tensor(a, device=device)
            fixed_frames[name] = decode(vae, rollout(model, context, seq, horizon))
        for key in action_diff:
            x, y = key.split("_vs_")
            action_diff[key].append(((fixed_frames[x] - fixed_frames[y]) ** 2).mean((2, 3, 4)).cpu())

        if b <= example_pos < b + batch_size:
            j = example_pos - b
            example = dict(true=true_frames[j].cpu(), model=pred_frames[j].cpu(), last=last_frame[j].cpu(),
                           **{k: v[j].cpu() for k, v in fixed_frames.items()})

    mse = {k: torch.cat(v) for k, v in mse.items()}           # (N, horizon)
    acc = {k: torch.cat(v) for k, v in acc.items()}
    action_diff = {k: torch.cat(v) for k, v in action_diff.items()}


    moving = mse["copy"][:, -1] >= mse["copy"][:, -1].median()

    steps = [k for k in [1, 2, 3, 5, 10, 15] if k <= horizon]
    report = {"num_rollouts": len(starts), "valid_episodes": valid_episodes.tolist(), "steps": steps}
    print(f"\n{'step':>4} | {'MSE model':>9} {'copy':>9} {'floor':>9} | {'moving: model':>13} {'copy':>9} | {'tok acc model':>13} {'copy':>6}")
    for k in steps:
        row = {
            "mse_model": mse["model"][:, k - 1].mean().item(),
            "mse_copy": mse["copy"][:, k - 1].mean().item(),
            "mse_recon_floor": mse["recon_floor"][:, k - 1].mean().item(),
            "moving_mse_model": mse["model"][moving, k - 1].mean().item(),
            "moving_mse_copy": mse["copy"][moving, k - 1].mean().item(),
            "acc_model": acc["model"][:, k - 1].mean().item(),
            "acc_copy": acc["copy"][:, k - 1].mean().item(),
        }
        report[f"step_{k}"] = row
        print(f"{k:>4} | {row['mse_model']:9.5f} {row['mse_copy']:9.5f} {row['mse_recon_floor']:9.5f} | "
              f"{row['moving_mse_model']:13.5f} {row['moving_mse_copy']:9.5f} | {row['acc_model']:13.4f} {row['acc_copy']:6.4f}")


    report["action_diff_at_horizon"] = {k: v[:, -1].mean().item() for k, v in action_diff.items()}
    print("\n행동별 예측 차이 (horizon 시점 픽셀 MSE):")
    for k, v in report["action_diff_at_horizon"].items():
        print(f"  {k:>14}: {v:.5f}")

    with open(f"{out_prefix}_metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    # 1) 스텝별 MSE 곡선
    x = np.arange(1, horizon + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(x, mse["model"].mean(0), marker="o", label="World Model (rollout)")
    plt.plot(x, mse["copy"].mean(0), marker="o", label="Copy last frame")
    plt.plot(x, mse["recon_floor"].mean(0), linestyle="--", label="VQ-VAE recon floor")
    plt.plot(x, mse["model"][moving].mean(0), marker=".", alpha=0.6, label="World Model (moving)")
    plt.plot(x, mse["copy"][moving].mean(0), marker=".", alpha=0.6, label="Copy (moving)")
    plt.xlabel("Prediction step")
    plt.ylabel("Pixel MSE")
    plt.title("Rollout error vs horizon")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{out_prefix}_mse.png", dpi=120)
    plt.close()

    # 2) 롤아웃 예시: 행 = 실제 / 모델 / 복사 / 고정 행동, 열 = 예측 스텝
    rows = [("Ground truth", example["true"]), ("Model (real actions)", example["model"]),
            ("Copy last frame", example["last"][None].expand(horizon, -1, -1, -1))]
    rows += [(f"Model (all {k})", example[k]) for k in ["left", "right", "gas"]]
    fig, axes = plt.subplots(len(rows), len(steps), figsize=(2.2 * len(steps), 2.3 * len(rows)))
    for r, (label, imgs) in enumerate(rows):
        for c, k in enumerate(steps):
            ax = axes[r, c]
            ax.imshow(imgs[k - 1].permute(1, 2, 0).numpy())
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"t+{k}")
            if c == 0:
                ax.set_ylabel(label, fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_example.png", dpi=120)
    plt.close()
    print(f"\n저장: {out_prefix}_metrics.json, {out_prefix}_mse.png, {out_prefix}_example.png")
    return report


if __name__ == "__main__":
    evaluate()
