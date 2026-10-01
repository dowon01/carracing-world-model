import matplotlib.pyplot as plt
import numpy as np

loaded_data = np.load('racing_data/play_action_frames.npy')
# loaded_data = np.load('racing_data/manual_frames.npy')
for i in range(5000):
    plt.imshow(loaded_data[i])    # 프레임 시각화
    plt.show()
    plt.close()

# save_path = 'racing_data/visualized_frame_1999.png'
# plt.savefig(save_path)
# plt.show()