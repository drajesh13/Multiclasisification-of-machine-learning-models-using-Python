import numpy as np


class DecisionTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y, depth=0):
        if len(y) == 0:
            self.tree = {"class": None, "is_leaf": True}
            return

        num_samples, num_features = X.shape
        num_classes = len(np.unique(y))

        # If only one class or max depth reached, create a leaf node with the majority class
        if num_classes == 1 or depth == self.max_depth:
            unique, counts = np.unique(y, return_counts=True)
            self.tree = {"class": unique[np.argmax(counts)], "is_leaf": True}
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
            unique, counts = np.unique(y, return_counts=True)
            self.tree = {"class": unique[np.argmax(counts)], "is_leaf": True}
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
        if self.tree["is_leaf"]:
            if self.tree["class"] is None:
                return np.zeros(
                    X.shape[0], dtype=int
                )  # or some other default prediction
            else:
                return np.full(X.shape[0], self.tree["class"])
        else:
            left_mask = X[:, self.tree["feature_index"]] <= self.tree["value"]
            right_mask = ~left_mask

            predictions = np.zeros(X.shape[0], dtype=int)
            predictions[left_mask] = self.tree["left"].predict(X[left_mask])
            predictions[right_mask] = self.tree["right"].predict(X[right_mask])

            return predictions

    def calculate_gini(self, labels):
        unique_classes, class_counts = np.unique(labels, return_counts=True)
        class_probabilities = class_counts / len(labels)
        gini = 1 - np.sum(class_probabilities**2)
        return gini


class XGBooster:
    def __init__(self, learning_rate=0.01, n_estimators=100, max_depth=3):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        num_samples, num_features = X.shape
        num_classes = len(np.unique(y))
        self.num_classes = num_classes

        self.trees = []

        for class_index in range(self.num_classes):
            y_binary = np.where(y == class_index, 1, -1)
            class_trees = []
            residuals = np.abs(y_binary).astype(float)

            for _ in range(self.n_estimators):
                tree = DecisionTree(max_depth=self.max_depth)
                tree.fit(X, y_binary)
                tree_preds = tree.predict(X).astype(float)
                residuals -= self.learning_rate * tree_preds
                residuals = np.abs(residuals)
                class_trees.append(tree)

            self.trees.append(class_trees)

    def get_params(self, deep=True):
        return {
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

    def predict(self, X):
        all_class_predictions = []

        for class_trees in self.trees:
            class_predictions = np.sum(
                [tree.predict(X) for tree in class_trees], axis=0
            )
            all_class_predictions.append(class_predictions)

        all_class_predictions = np.array(all_class_predictions)
        return np.argmax(all_class_predictions, axis=0)


# from sklearn.model_selection import GridSearchCV

# # # Example usage:
# X_train = np.array([[1, 1], [0.2, 0.2], [33, 3.3], [4, 44], [0, 55]])
# y_train = np.array([0, 0, 1, 2, 2])

# param_test1 = {
#  'max_depth':range(3,10,2),
# }
# cv_params = {'max_depth': [1,2,3,4,5,6]}    # parameters to be tries in the grid search
# fix_params = {'learning_rate': 0.01, 'n_estimators': 100}   #other parameters, fixed for the moment
# csv = GridSearchCV(estimator=XGBooster(**fix_params), scoring='accuracy', param_grid=cv_params, cv=5, n_jobs=3)
# csv.fit(X_train,y_train)
# print(csv.best_params_, csv.best_score_)

# xgb = XGBooster()
# xgb.fit(X_train, y_train)

# X_test = np.array([[1, 1], [0.2, 0.2], [33, 3.3], [4, 44], [0, 55]])
# predictions = xgb.predict(X_test)
# print(predictions)
