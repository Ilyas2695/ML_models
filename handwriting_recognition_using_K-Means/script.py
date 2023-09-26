import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

digits = datasets.load_digits()
print(digits.data)
print("---------------------")
print(digits.target)

plt.gray()
plt.matshow(digits.images[100])
plt.show()

print(digits.target)

model = KMeans(n_clusters=10, random_state=42)

model.fit(digits.data)

fig = plt.figure(figsize=(8,3))
fig.suptitle('Cluster Center Images', fontsize=14, fontweight='bold')

for i in range(10):
 
  # Initialize subplots in a grid of 2X5, at i+1th position
  ax = fig.add_subplot(2, 5, 1 + i)
 
  # Display images
  ax.imshow(model.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)

plt.show()

new_samples = np.array([
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.53,0.76,0.76,0.15,0.00,0.00,0.00,0.00,7.40,7.62,7.62,6.64,0.92,0.00,0.00,0.00,1.83,2.29,2.82,7.62,3.05,0.00,0.00,0.00,0.00,0.00,4.04,7.62,2.29,0.00,0.00,0.00,1.60,6.03,7.62,4.73,0.00,0.00,0.00,1.15,7.55,7.62,7.55,6.79,6.64,0.46,0.00,0.54,5.11,5.34,5.34,4.66,3.59,0.15],
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.08,2.21,2.29,1.30,0.00,0.00,0.00,0.00,2.75,7.62,7.62,7.55,3.36,0.00,0.00,0.00,4.27,7.24,1.22,5.80,7.55,0.00,0.00,0.00,4.57,6.10,0.00,3.51,7.62,0.00,0.00,0.00,4.57,6.79,0.99,4.80,6.63,0.00,0.00,0.00,2.06,7.32,7.62,7.47,5.26,0.00,0.00,0.00,0.00,1.07,3.28,3.81,1.83,0.00],
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,1.22,3.96,4.57,4.57,2.98,0.00,0.00,0.00,5.11,7.55,6.18,6.86,7.62,0.00,0.00,0.00,0.31,0.46,1.91,6.79,7.17,0.00,0.00,0.00,0.84,4.88,7.55,6.64,1.45,0.00,0.00,1.37,7.09,7.62,7.62,5.64,5.34,2.21,0.00,2.60,7.62,7.62,6.56,5.49,5.34,2.21,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
[0.00,0.00,0.00,0.76,1.30,0.00,0.00,0.00,0.00,0.00,1.91,7.17,7.62,7.02,4.50,0.61,0.00,2.14,7.40,6.41,2.60,5.34,7.55,5.57,0.00,6.71,6.33,0.46,0.00,0.00,4.73,6.10,0.00,7.62,3.35,0.00,0.00,0.00,4.57,6.10,0.00,7.40,4.35,0.00,0.00,0.00,5.87,5.79,0.00,5.03,7.55,5.11,3.20,3.36,7.47,4.27,0.00,0.46,4.34,6.79,7.62,7.62,6.56,0.92]
])

new_labels = model.predict(new_samples)
print(new_labels)

for i in range(len(new_labels)):
  if new_labels[i] == 0:
    print(0, end='')
  elif new_labels[i] == 1:
    print(9, end='')
  elif new_labels[i] == 2:
    print(2, end='')
  elif new_labels[i] == 3:
    print(1, end='')
  elif new_labels[i] == 4:
    print(6, end='')
  elif new_labels[i] == 5:
    print(8, end='')
  elif new_labels[i] == 6:
    print(4, end='')
  elif new_labels[i] == 7:
    print(5, end='')
  elif new_labels[i] == 8:
    print(7, end='')
  elif new_labels[i] == 9:
    print(3, end='')



