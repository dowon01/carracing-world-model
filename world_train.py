from world_transformer import WorldTransformer, WorldSequenceDataset, load_episode_ids
from config import TOKENS_PATH, ACTIONS_PATH, EPISODES_PATH, WORLD_MODEL_PATH, WORLD_MODEL_CONFIG, get_device
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


def split_episodes(valid_ratio=0.1, seed=0):
    # 에피소드 단위로 학습/검증 분할 (겹치는 윈도우가 train/valid에 동시에 들어가는 누출 방지)
    num_frames = len(np.load(TOKENS_PATH, mmap_mode="r"))
    episodes = np.unique(load_episode_ids(EPISODES_PATH, num_frames))
    rng = np.random.default_rng(seed)
    rng.shuffle(episodes)
    num_valid = max(1, int(round(valid_ratio * len(episodes))))
    return episodes[num_valid:], episodes[:num_valid] # train, valid


def run_epoch(model, loader, criterion, device, optimizer=None, desc=""):
    # optimizer가 있으면 학습, 없으면 검증
    training = optimizer is not None
    model.train(training)
    num_tokens = model.num_tokens

    total_loss, correct, copy_correct, changed_correct, changed_total, count = 0.0, 0, 0, 0, 0, 0
    pbar = tqdm(loader, desc=desc, unit="batch")

    with torch.set_grad_enabled(training):
        for batch in pbar:
            states = batch["states"].to(device) # (B, seq_len, 256)
            actions = batch["actions"].to(device) # (B, seq_len, 3)
            targets = batch["targets"].to(device) # (B, seq_len, 256)

            # 예측 (logits 모양 : B, T, 256, num_tokens)
            logits = model(states, actions)
            loss = criterion(logits.reshape(-1, num_tokens), targets.reshape(-1))

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            # 정확도: 모델 vs "직전 프레임 복사" 베이스라인, 그리고 실제로 바뀐 토큰에 대한 정확도
            preds = logits.argmax(-1)
            changed = targets != states
            total_loss += loss.item() * targets.numel()
            correct += (preds == targets).sum().item()
            copy_correct += (~changed).sum().item()
            changed_correct += ((preds == targets) & changed).sum().item()
            changed_total += changed.sum().item()
            count += targets.numel()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    return {
        "loss": total_loss / count,
        "acc": correct / count,
        "copy_acc": copy_correct / count,
        "changed_acc": changed_correct / max(changed_total, 1),
    }


def train_world_model(num_epochs=10, batch_size=16, learning_rate=3e-4, save_path=WORLD_MODEL_PATH, max_batches=None,
                      resume=False, epochs_this_run=None):
    device = get_device()
    print(f"학습 시작 : {device} 사용")
    seq_len = WORLD_MODEL_CONFIG["seq_len"]

    train_episodes, valid_episodes = split_episodes()
    print(f"train episodes: {sorted(train_episodes.tolist())}, valid episodes: {sorted(valid_episodes.tolist())}")

    train_dataset = WorldSequenceDataset(TOKENS_PATH, ACTIONS_PATH, seq_len, EPISODES_PATH, train_episodes)
    valid_dataset = WorldSequenceDataset(TOKENS_PATH, ACTIONS_PATH, seq_len, EPISODES_PATH, valid_episodes)
    if max_batches is not None: # 빠른 동작 확인용
        train_dataset.starts = train_dataset.starts[: max_batches * batch_size]
        valid_dataset.starts = valid_dataset.starts[: max_batches * batch_size]
    print(f"train samples: {len(train_dataset)}, valid samples: {len(valid_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # 모델, 손실 함수, 옵티마이저
    model = WorldTransformer(**WORLD_MODEL_CONFIG).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss()
    print(f"파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

    train_history = []
    valid_history = []
    best_valid_loss = float("inf")
    start_epoch = 0

    # 매 epoch 학습 상태를 저장해 두고, 중단되면 --resume으로 이어서 학습
    state_path = save_path.replace(".pth", "_state.pth")
    if resume and os.path.exists(state_path):
        state = torch.load(state_path, map_location=device)
        model.load_state_dict(state["model"])
        if state.get("optimizer") is not None:
            optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"]
        train_history, valid_history = state["train_history"], state["valid_history"]
        best_valid_loss = state["best_valid_loss"]
        print(f"이어서 학습: {start_epoch} epoch 완료 상태에서 시작")

    end_epoch = num_epochs if epochs_this_run is None else min(num_epochs, start_epoch + epochs_this_run)
    for epoch in range(start_epoch, end_epoch):
        train_stats = run_epoch(model, train_loader, criterion, device, optimizer, desc=f"Epoch {epoch+1}/{num_epochs} [train]")
        valid_stats = run_epoch(model, valid_loader, criterion, device, desc=f"Epoch {epoch+1}/{num_epochs} [valid]")
        scheduler.step()

        train_history.append(train_stats["loss"])
        valid_history.append(valid_stats["loss"])

        print(f"Epoch [{epoch+1}] 완료 - Train Loss: {train_stats['loss']:.4f}, Valid Loss: {valid_stats['loss']:.4f}")
        print(f"  Valid Acc: {valid_stats['acc']:.4f} (copy baseline: {valid_stats['copy_acc']:.4f}), "
              f"Changed-token Acc: {valid_stats['changed_acc']:.4f}")

        # 검증 손실이 가장 좋을 때만 모델 저장
        if valid_stats["loss"] < best_valid_loss:
            best_valid_loss = valid_stats["loss"]
            torch.save(model.state_dict(), save_path)
            print(f"  모델 저장: {save_path}")

        torch.save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "epoch": epoch + 1, "train_history": train_history, "valid_history": valid_history,
            "best_valid_loss": best_valid_loss,
        }, state_path)

    if end_epoch < num_epochs:
        print(f"{end_epoch}/{num_epochs} epoch까지 학습 후 중단 (--resume으로 이어서 학습)")
        return

    # 손실 그래프 저장
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_history) + 1), train_history, marker="o", label="Train Loss")
    plt.plot(range(1, len(valid_history) + 1), valid_history, marker="o", label="Valid Loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path.replace(".pth", "_loss.png"))
    plt.close()

    print("학습 완료")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--resume", action="store_true", help="저장된 학습 상태에서 이어서 학습")
    parser.add_argument("--epochs-this-run", type=int, default=None, help="이번 실행에서 학습할 최대 epoch 수")
    args = parser.parse_args()
    train_world_model(num_epochs=args.epochs, resume=args.resume, epochs_this_run=args.epochs_this_run)
