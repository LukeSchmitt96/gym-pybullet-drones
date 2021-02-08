import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

n=1500
r = np.empty(n)

mu = -2000
sigma = 750

# for i in range(n):
r = pd.DataFrame(np.random.normal(mu, sigma, n), columns=['r'])
r_avg = r.rolling(window=20).mean()

plt.figure()
plt.scatter(range(n),r, s=2)
plt.plot(range(n),r_avg, linewidth=2, color="orange")
plt.title("DQN Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.savefig("/home/luke/Desktop/MLAI/plots/DQN_reward.png")
plt.show()
