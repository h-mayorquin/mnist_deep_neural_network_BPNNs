import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

# Load data

percentage = 0.1
percentage = ''
folder = './data/'
name = 'information_distances'
format = '.npy'
file_name = folder + name + str(percentage) + format
distances = np.load(file_name)

dimensions = np.arange(10, 100, 10)
stress_vector = np.zeros_like(dimensions)

for i, dim in enumerate(dimensions):

    # Define classifier
    n_comp = dim
    max_iter = 3000
    eps = 1e-9
    mds = MDS(n_components=n_comp, max_iter=max_iter, eps=eps,
              n_jobs=3, dissimilarity='precomputed')

    x = mds.fit(distances)
    stress = x.stress_
    print 'The stress is', stress
    stress_vector[i] = stress

# Plot Here

plt.plot(dimensions, stress_vector, '*')
plt.show()
