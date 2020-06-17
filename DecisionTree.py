from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import numpy as np

features = []
f = np.load('./FEATURES_v2.npy')
for i in range(len(f)):
	features.append(f[i])

#print(features[len(features)-1].shape)

X, Y = zip(*features)
X = np.array(X).reshape(-1, 32)
Y = np.array(Y).reshape(-1)
#print(X.shape)

tree = DecisionTreeClassifier(criterion = 'entropy', max_depth = 10)
tree.fit(X, Y)
#print(tree.predict(X[0].reshape(1, -1)))
#print(tree.score(X, Y))


SVM = SVC(kernel = 'sigmoid', gamma = 'auto', probability = True)
SVM.fit(X, Y)
