import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOG_FILE_PATH = 'logs/training_log_21-09-2026_07-53-37.csv'

df = pd.read_csv(LOG_FILE_PATH)
print(df.tail())

df.plot(x='epoch', y=['loss', 'val_loss'], kind='line')


plt.title("Loss chart")
plt.savefig("analysis/charts/loss_chart-21-09-2026-50epok")