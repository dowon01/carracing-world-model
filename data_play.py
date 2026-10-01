import gymnasium as gym
import numpy as np
import cv2
import pygame
from pygame.locals import K_LEFT, K_RIGHT, K_UP, K_DOWN

def collect_manual_data(save_path='racing_data/play_action_frames.npy', action_save_path='racing_data/play_actions.npy', episode_save_path='racing_data/play_episode_ids.npy', target_frames=10000):
    env = gym.make("CarRacing-v3", render_mode="human")
    obs, _ = env.reset()
    
    frames = []
    actions = []
    episode_ids = []  # 각 프레임이 속한 에피소드 번호 (시퀀스가 에피소드 경계를 넘지 않도록)
    episode = 0
    clock = pygame.time.Clock()
    
    print(f"수동 운전 시작! (목표: {target_frames} 프레임)")

    running = True
    while len(frames) < target_frames and running:
        action = np.array([0.0, 0.0, 0.0])
        
        pygame.event.pump()
        keys = pygame.key.get_pressed()
        
        if keys[K_LEFT]:  action[0] = -1.0
        if keys[K_RIGHT]: action[0] = 1.0
        if keys[K_UP]:    action[1] = 0.5
        if keys[K_DOWN]:  action[2] = 0.8

        # 1. 액션 먼저 기록 (저장 규칙: frames[i]는 actions[i]를 실행한 "결과" 관측값)
        actions.append(action.copy())

        # 2. 환경 실행
        obs, reward, terminated, truncated, _ = env.step(action)
        
        # 3. 이미지 전처리 및 저장
        frame = cv2.resize(obs, (64, 64))
        frames.append(frame)
        episode_ids.append(episode)
        
        if len(frames) % 1000 == 0:
            print(f"현재 수집된 프레임: {len(frames)} / {target_frames}")

        if terminated or truncated:
            obs, _ = env.reset()
            episode += 1
            
        clock.tick(30)

    # 4. 데이터 저장 (이미지와 액션을 각각 저장)
    if len(frames) > 0:
        np.save(save_path, np.array(frames, dtype=np.uint8))
        np.save(action_save_path, np.array(actions, dtype=np.float32)) # 액션 저장 추가
        np.save(episode_save_path, np.array(episode_ids, dtype=np.int64)) # 에피소드 번호 저장
        print(f"이미지 저장 완료: {save_path} ({np.array(frames).shape})")
        print(f"액션 저장 완료: {action_save_path} ({np.array(actions).shape})")
    
    env.close()
    pygame.quit()

if __name__ == "__main__":
    collect_manual_data()