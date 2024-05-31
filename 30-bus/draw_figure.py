import numpy as np
import matplotlib.pyplot as plt

pmax = [80, 80, 50, 55, 30, 40]
error_M1 = np.load('./results/M1_errors.npz', allow_pickle=True)
error_M1 = error_M1['errors']
x_m1 = list(range(2000))
y_m1 = np.mean(error_M1, axis=-1)

error_M2 = np.load('./results/M2_errors.npz', allow_pickle=True)
error_M2 = error_M2['errors']
x_m2 = np.array(list(range(2000))) + 2000
y_m2 = np.mean(error_M2, axis=-1)

error_M3 = np.load('./results/M3_errors.npz', allow_pickle=True)
error_M3 = error_M3['errors']
x_m3 = np.array(list(range(2000))) + 2000 * 2
y_m3 = np.mean(error_M3, axis=-1)

error_M4 = np.load('./results/M4_errors.npz', allow_pickle=True)
error_M4 = error_M4['errors']
x_m4 = np.array(list(range(2000))) + 2000 * 3
y_m4 = np.mean(error_M4, axis=-1)

error_M5 = np.load('./results/M5_old_errors.npz', allow_pickle=True)
error_M5 = error_M5['errors']
x_m5 = np.array(list(range(2000))) + 2000 * 4
y_m5 = np.mean(error_M5, axis=-1)


plt.scatter(x_m1, y_m1, c='darkorange', s=10)
plt.scatter(x_m2, y_m2, c='forestgreen', s=10)
plt.scatter(x_m3, y_m3, c='purple', s=10)
plt.scatter(x_m4, y_m4, c='lightcoral', s=10)
plt.scatter(x_m5, y_m5, c='firebrick', s=10)

x_line = np.array(list(range(2000*5)))
y_line = np.ones((2000*5, ))
plt.plot(x_line, y_line, c='black', linewidth=1)
plt.legend(['M1', 'M2', 'M3', 'M4', 'M5'])
plt.ylabel("Errors of PG (MW)")
plt.xlabel("Sample index of different methods")
plt.xticks([])
plt.yticks([0, 1, 5, 10, 15, 20, 25])

plt.show()
print(123)