import numpy as np

class SVM:
    def __init__(self, C=1.0):
        # SVM initialization with regularization parameter C
        self.C = C
        self.w = None  # Weight vector
        self.b = None  # Bias term

    def hinge_loss(self, w, b, x, y):
        # Hinge loss function for SVM
        reg = 0.5 * np.dot(w, w)  # Regularization term
        opt_term = y * (np.dot(w, x) + b)  # Optimization term
        loss = reg + self.C * max(0, 1 - opt_term)  # Hinge loss
        return loss

    def fit(self, X, Y, batch_size=100, learning_rate=0.001, epochs=5):
        # Training the SVM model using Stochastic Gradient Descent
        num_features = X.shape[1]  # Number of features in the dataset
        num_samples = X.shape[0]  # Number of samples in the dataset
        self.w = np.zeros(num_features)  # Initialize weights to zeros
        self.b = 0  # Initialize bias to zero
        losses = []  # To store the loss at each epoch


        for epoch in range(epochs):
            for batch_start in range(0, num_samples, batch_size):
                batch_end = min(batch_start + batch_size, num_samples)
                X_batch = X[batch_start:batch_end]
                Y_batch = Y[batch_start:batch_end]
                grad_w = 0  # Gradient of weights
                grad_b = 0  # Gradient of bias

                for i in range(len(X_batch)):
                    ti = Y_batch.iloc[i] * (np.dot(self.w, X_batch.iloc[i]) + self.b)

                    if ti > 1:
                        grad_w += 0  # No update to weights if ti > 1
                        grad_b += 0  # No update to bias if ti > 1
                    else:
                        grad_w += self.C * Y_batch.iloc[i] * X_batch.iloc[i]
                        grad_b += self.C * Y_batch.iloc[i]

                # Update weights and bias using Stochastic Gradient Descent
                self.w = self.w - learning_rate * self.w + learning_rate * grad_w
                self.b = self.b + learning_rate * grad_b

            # Calculate total loss for the current epoch
            total_loss = np.sum([self.hinge_loss(self.w, self.b, X.iloc[i], Y.iloc[i]) for i in range(len(X))])
            losses.append(total_loss)

        return self.w, self.b, losses

    def predict(self, X):
        # Make predictions using the trained SVM model
        predictions = np.dot(X, self.w) + self.b
        return np.sign(predictions)  # Return the sign of the predictions (1 or -1)
