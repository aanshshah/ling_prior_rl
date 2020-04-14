import numpy as np
import matplotlib.pyplot as plt


first_run_data = np.load('dqn_first_try.npy')
# second_run_data = np.load('second_run_episode_rewards.npy')


plt.plot(np.arange(len(first_run_data)),first_run_data)
plt.show()



# plt.plot(np.arange(len(second_run_data)),second_run_data)
# plt.show()


