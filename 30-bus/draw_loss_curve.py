import numpy as np
import matplotlib.pyplot as plt
import csv

csv_reader = csv.reader(open("./models/Reconstruct/losses.csv"))
data_all = []
index = 0
for line in csv_reader:
    if index == 0:
        index += 1
        continue
    if len(line)==0:
        continue
    data_all.append([float(line[0]), float(line[1]), float(line[2]), float(line[3])])

data_all = np.array(data_all)
plt.plot(data_all[:, 0], data_all[:, 1], linewidth=1, linestyle='-', color='tomato', label="Train error in 34.28% penetration rate")
plt.plot(data_all[:, 0], data_all[:, 2], linewidth=1, linestyle='-', color='green', label="Testing error in 34.28% penetration rate")
plt.plot(data_all[:, 0], data_all[:, 3], linewidth=1, linestyle='-', color='red', label="Testing error in 39.51% penetration rate")
plt.legend()
plt.ylim(0,0.03)
plt.ylabel('Mean Absolute Error')
plt.xlabel('Number of Epochs')
plt.show()
print(123)