import numpy as np


class MLP:
    def __init__(self, num_features, hidden_size, num_classes):
        # The object initializes and remembers its own weights and biases
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self._init_weights()

    def _init_weights(self):
        """ initialization, suited for ReLU hidden units, including the weights and bias"""
        self.W1 = np.random.randn(self.num_features, self.hidden_size) * np.sqrt(2.0 / self.num_features)
        self.b1 = np.zeros(self.hidden_size)
        self.W2 = np.random.randn(self.hidden_size, self.num_classes) * np.sqrt(2.0 / self.hidden_size)
        self.b2 = np.zeros(self.num_classes)

    # Functions for calculations
    def relu(self, z):
        """ Applies ReLU"""
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def softmax(self, logits):
        " Applies softmax function"
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_values = np.exp(logits)
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def forward(self, X):
        """ Hidden layer: XW1 + b1, then ReLU"""
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)

        """ Output layer: a1 W2 + b2, then softmax"""
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2

    def compute_loss(self, probs, y_true):
        """ Calculates the loss, use epsilon to avoid possible undenfined values"""
        epsilon = 1e-8
        probs = np.clip(probs, epsilon, 1.0 - epsilon)
        correct_class_probs = np.sum(y_true * probs, axis=1)
        return -np.mean(np.log(correct_class_probs))

    def compute_gradients(self, X, y_true, probs):
        """ Backpropagation: output layer error, then propagated back through ReLU """
        n = X.shape[0]
        d_z2 = (probs - y_true) / n
        weight_gradients_2 = np.dot(self.a1.T, d_z2)
        bias_gradients_2 = np.sum(d_z2, axis=0)

        d_a1 = np.dot(d_z2, self.W2.T)
        d_z1 = d_a1 * self.relu_derivative(self.z1)
        weight_gradients_1 = np.dot(X.T, d_z1)
        bias_gradients_1 = np.sum(d_z1, axis=0)

        return weight_gradients_1, bias_gradients_1, weight_gradients_2, bias_gradients_2

    def update_weights(self, weight_gradients_1, bias_gradients_1, weight_gradients_2, bias_gradients_2, learning_rate):
        """ Update the weights and bias based on the learning rate and gradient"""
        self.W1 -= learning_rate * weight_gradients_1
        self.b1 -= learning_rate * bias_gradients_1
        self.W2 -= learning_rate * weight_gradients_2
        self.b2 -= learning_rate * bias_gradients_2

    def compute_accuracy(self, probs, y_true):
        """ Computes the accuracy of the predictions to the true labels"""
        preds = np.argmax(probs, axis=1)
        if y_true.ndim > 1:
            y_true = np.argmax(y_true, axis=1)
        return np.mean(preds == y_true)

    def get_weight_and_bias(self):
        """ Used for saving the best model during early stopping"""
        return self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy()

    def set_weights(self, best_weights):
        """ Used for restoring the best model"""
        self.W1, self.b1, self.W2, self.b2 = [w.copy() for w in best_weights]

    def reset_weights(self):
        """  Reset to a fresh random initialization for a new hyperparameter configuration"""
        self._init_weights()

    # Flow

    def training(self, X_train, y_train, learning_rate):
        """
        Trains the MLP for one full epoch.
        """
        # 1. Compute predictions (Forward pass handles hidden layer + output layer + softmax)
        probs = self.forward(X_train)

        # 2. Compute accuracy
        training_accuracy = self.compute_accuracy(probs, y_train)

        # 3. Compute loss
        training_loss = self.compute_loss(probs, y_train)

        # 4. Compute gradients (backpropagation)
        weight_gradients_1, bias_gradients_1, weight_gradients_2, bias_gradients_2 = self.compute_gradients(X_train, y_train, probs)

        # 5. Update weights using gradient descent
        self.update_weights(weight_gradients_1, bias_gradients_1, weight_gradients_2, bias_gradients_2, learning_rate)

        # 6. Return loss and accuracy
        return training_loss, training_accuracy

    def validation(self, X_val, y_val):
        """
        Validates the model and computes accuracy.
        """
        # 1. Compute predictions
        probs = self.forward(X_val)

        # 2. Compute accuracy
        validation_accuracy = self.compute_accuracy(probs, y_val)

        # 3. Compute loss
        validation_loss = self.compute_loss(probs, y_val)

        # 4. Return loss and accuracy
        return validation_loss, validation_accuracy

    def testing(self, X_test, y_test):
        """
        Tests the model on unseen data and computes accuracy.
        """

        # 1. Compute predictions
        probs = self.forward(X_test)

        # 2. Compute accuracy
        testing_accuracy = self.compute_accuracy(probs, y_test)

        # 3. Compute loss
        testing_loss = self.compute_loss(probs, y_test)

        # 4. Return loss and accuracy
        return testing_loss, testing_accuracy
