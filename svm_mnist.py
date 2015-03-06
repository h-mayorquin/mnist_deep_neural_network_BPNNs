from sklearn.svm import SVC
from load_files import training_ims, training_labels
import numpy as np

# Transform everything into array
training_ims = np.array(training_ims)
training_labels = np.array(training_labels)

# Select quantity of data to use
N_data_total = len(training_labels)
percentage = 0.05
N_to_use = int(percentage * N_data_total)

# Decide how much data to use
X = training_ims[0:N_to_use]
Y = training_labels[0:N_to_use]

# Implement the support vector machine
clf = SVC()
clf.fit(X, Y)
score = clf.score(X, Y) * 100
print 'The score is = ', score
