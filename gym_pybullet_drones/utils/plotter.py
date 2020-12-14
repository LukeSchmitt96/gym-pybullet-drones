import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("/content/PPO_reward - monitor.csv")

t = data['t']; r = data['r']

r_avg = r.rolling(window=20).mean()

plt.figure()
plt.scatter(t,r, s=2)
plt.plot(t,r_avg, linewidth=2, color="orange")
plt.xlim([t.min(), t.max()])
plt.grid()
plt.xlabel("Timestep")
plt.ylabel("Reward")
plt.title("PPO Reward")
plt.savefig("PPO_reward.png")
plt.show()