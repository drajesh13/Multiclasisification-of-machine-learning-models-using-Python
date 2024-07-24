import numpy as np
from sklearn.decomposition import PCA

class SVM:
    def __init__(self, C=1.0, n_components=None):
        self.C = C
        self.classifiers = None
        self.pca = PCA(n_components=n_components)

    def hinge_loss(self, w, b, x, y):
        reg = 0.5 * np.dot(w, w)
        opt_term = y * (np.dot(w, x) + b)
        loss = reg + self.C * max(0, 1 - opt_term)
        return loss

    def fit_one_vs_rest(self, X, Y, batch_size=100, learning_rate=0.001, epochs=5):
        num_features = X.shape[1]
        num_samples = X.shape[0]
        w = np.zeros(num_features)
        b = 0
        losses = []

        for epoch in range(epochs):
            for batch_start in range(0, num_samples, batch_size):
                batch_end = min(batch_start + batch_size, num_samples)
                X_batch = X[batch_start:batch_end]
                Y_batch = Y[batch_start:batch_end]
                grad_w = 0
                grad_b = 0

                for i in range(len(X_batch)):
                    ti = Y_batch[i] * (np.dot(w, X_batch[i]) + b)

                    if ti > 1:
                        grad_w += 0
                        grad_b += 0
                    else:
                        grad_w += self.C * Y_batch[i] * X_batch[i]
                        grad_b += self.C * Y_batch[i]

                w = w - learning_rate * w + learning_rate * grad_w
                b = b + learning_rate * grad_b

            total_loss = np.sum([self.hinge_loss(w, b, X[i], Y[i]) for i in range(len(X))])
            losses.append(total_loss)

        return w, b, losses

    def fit(self, X, Y, batch_size=100, learning_rate=0.001, epochs=10):
        X = self.pca.fit_transform(X)
        self.classifiers = {}
        for class_label in np.unique(Y):
            Y_binary = np.where(Y == class_label, 1, -1)
            w, b, losses = self.fit_one_vs_rest(X, Y_binary, batch_size, learning_rate, epochs)
            self.classifiers[class_label] = (w, b)

    def predict(self, X):
        # Apply the same PCA transformation to the test data
        X = self.pca.transform(X)
        
        predictions = np.zeros((X.shape[0], len(self.classifiers)))
        for class_label, (w, b) in self.classifiers.items():
            print(class_label)
            predictions[:, class_label] = np.dot(X, w) + b
        return np.argmax(predictions, axis=1)
    
# X_train = np.array([[1, 1], [0.2, 0.2], [33, 3.3], [4, 44], [0, 55]]) 
# y_train = np.array([0, 0, 1, 2, 2])

# svm_model = SVM() 
# svm_model.fit(X_train, y_train)

# X_test = np.array([[1, 1], [0.2, 0.2], [33, 3.3], [4, 44], [0, 55]]) 
# predictions = svm_model.predict(X_test) 
# print(predictions)