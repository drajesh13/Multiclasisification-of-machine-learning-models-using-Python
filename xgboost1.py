import numpy as np
from scipy.stats import mode


class XGBooster:
    def __init__(self, learning_rate=0.1, n_estimators=100, max_depth=3):
        # XGBooster initialization
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []  # List to store individual trees

    def sigmoid(self, x):
        # Sigmoid activation function
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        # Training the XGBooster model
        num_samples, num_features = X.shape
        self.trees = []  # Reset trees for each fit

        # Initialize residuals to be the labels (for the first iteration)
        residuals = y.copy()

        for _ in range(self.n_estimators):
            # Create a tree for each iteration
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X, residuals)

            # Predictions of the current tree
            tree_preds = tree.predict(X)

            # Convert residuals to the same data type as labels before subtraction
            residuals = residuals.astype(tree_preds.dtype)

            # Update residuals with the negative gradient (residuals - learning_rate * gradient)
            residuals -= self.learning_rate * tree_preds

            # Add the tree to the list of trees
            self.trees.append(tree)

    def predict(self, X):
        # Make predictions using the trained XGBooster model
        all_tree_preds = np.sum([tree.predict(X) for tree in self.trees], axis=0)
        return self.softmax(
            all_tree_preds
        )  # Apply softmax to the sum of tree predictions

        # x = [tree.predict(X) for tree in self.trees]
        # all_tree_preds = np.array(
        #    [np.bincount(tree.predict(X)).argmax() for tree in self.trees]
        # )
        # all_tree_preds = np.sum([tree.predict(X) for tree in self.trees], axis=0)
        # return self.sigmoid(
        #    all_tree_preds
        # )  # Apply sigmoid to the sum of tree predictions


# Simple DecisionTree class for the XGBooster
class DecisionTree:
    def __init__(self, max_depth=3):
        # DecisionTree initialization
        self.max_depth = max_depth
        self.tree = None  # Tree structure to be built during training

    def fit(self, X, y, depth=0):
        # Training the DecisionTree model using recursive depth-first approach
        num_samples, num_features = X.shape
        num_classes = len(np.unique(y))

        # If only one class or max depth reached, create a leaf node with the majority class
        if num_classes == 1 or depth == self.max_depth:
            # self.tree = {"class": np.argmax(np.bincount(y)), "is_leaf": True}
            self.tree = {"class": mode(y).mode, "is_leaf": True}
            return

        # Find the best split based on Gini impurity
        best_gini = float("inf")
        best_split = None

        for feature_index in range(num_features):
            unique_values = np.unique(X[:, feature_index])

            for value in unique_values:
                left_mask = X[:, feature_index] <= value
                right_mask = ~left_mask

                gini_left = self.calculate_gini(y[left_mask])
                gini_right = self.calculate_gini(y[right_mask])

                weighted_gini = (len(y[left_mask]) / num_samples) * gini_left + (
                    len(y[right_mask]) / num_samples
                ) * gini_right

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_split = {"feature_index": feature_index, "value": value}

        # If no split improves Gini impurity, create a leaf node with the majority class
        if best_gini == float("inf"):
            self.tree = {"class": np.argmax(np.bincount(y)), "is_leaf": True}
            return

        # Split the data based on the best split
        left_data = X[X[:, best_split["feature_index"]] <= best_split["value"]]
        left_labels = y[X[:, best_split["feature_index"]] <= best_split["value"]]

        right_data = X[X[:, best_split["feature_index"]] > best_split["value"]]
        right_labels = y[X[:, best_split["feature_index"]] > best_split["value"]]

        # Recursively build left and right subtrees
        self.tree = {
            "feature_index": best_split["feature_index"],
            "value": best_split["value"],
            "is_leaf": False,
        }
        self.tree["left"] = DecisionTree(max_depth=self.max_depth)
        self.tree["left"].fit(left_data, left_labels, depth + 1)

        self.tree["right"] = DecisionTree(max_depth=self.max_depth)
        self.tree["right"].fit(right_data, right_labels, depth + 1)

    def predict(self, X):
        # Make predictions using the trained DecisionTree model
        # if self.tree["is_leaf"]:
        #     return np.full(X.shape[0], self.tree["class"])
        # else:
        #     left_mask = X[:, self.tree["feature_index"]] <= self.tree["value"]
        #     right_mask = ~left_mask

        #     predictions = np.zeros(X.shape[0])
        #     predictions[left_mask] = self.tree["left"].predict(X[left_mask])
        #     predictions[right_mask] = self.tree["right"].predict(X[right_mask])

        #     return predictions
        predictions = np.zeros(X.shape[0], dtype=int)
        self._traverse_tree(X, self.tree, predictions)
        return predictions

    def _traverse_tree(self, X, node, predictions):
        if node["is_leaf"]:
            predictions[:] = node["class"]
        else:
            left_mask = X[:, node.feature_index] <= node.value
            right_mask = ~left_mask

            self._traverse_tree(X[left_mask], node.left, predictions[left_mask])
            self._traverse_tree(X[right_mask], node.right, predictions[right_mask])

    def calculate_gini(self, labels):
        # Calculate Gini impurity for a set of labels
        unique_classes, class_counts = np.unique(labels, return_counts=True)
        class_probabilities = class_counts / len(labels)
        gini = 1 - np.sum(class_probabilities**2)
        return gini
