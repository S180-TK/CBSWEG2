import numpy as np


class MultinomialLogisticRegression:
    def __init__(self, num_features, num_classes):
        """ The object initializes its own weights and biases, and saves important variables"""
        self.num_features = num_features
        self.num_classes = num_classes
        self.weights = np.random.randn(self.num_features, self.num_classes) * 0.01
        self.bias = np.zeros(self.num_classes)

    # Functions for calculations
    def softmax(self, logits):
        """Calculate the Logits"""
        logits -= np.max(logits, axis=1, keepdims=True)
        exp_values = np.exp(logits)
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def forward(self, X):
        """ Calculates XW + b and applies softmax"""
        logits = np.dot(X, self.weights) + self.bias
        return self.softmax(logits)

    def compute_loss(self, probs, y_true):
        """ Calculates the loss, use epsilon to avoid possible undenfined values"""
        epsilon = 1e-8
        probs = np.clip(probs, epsilon, 1.0 - epsilon)
        correct_class_probs = np.sum(y_true * probs, axis=1)
        return -np.mean(np.log(correct_class_probs))

    def compute_gradients(self, X, y_true, probs):
        """ Computes the gradients of the loss """
        errors = probs - y_true
        weight_gradients = np.dot(X.T, errors) / X.shape[0]
        bias_gradients = np.sum(errors, axis=0) / X.shape[0]
        return weight_gradients, bias_gradients

    def update_weights(self, weight_gradients, bias_gradients, learning_rate):
        """ Update the weights and bias based on the learning rate and gradient"""
        self.weights -= (learning_rate * weight_gradients)
        self.bias -= (learning_rate * bias_gradients)

    def compute_accuracy(self, probs, y_true):
        """ Computes the accuracy of the predictions to the true labels"""
        preds = np.argmax(probs, axis=1)
        if y_true.ndim > 1:
            y_true = np.argmax(y_true, axis=1)
        return np.mean(preds == y_true)

    def get_weight_and_bias(self):
        """ Get the weights and bias of the model"""
        return self.weights.copy(), self.bias.copy()

    def set_weights(self, best_weights):
        """ Set the weights and bias of the model"""
        self.weights = best_weights[0]
        self.bias = best_weights[1]

    def reset_weights(self):
        """ Reset the weights and bias for a fresh start"""
        self.weights = np.random.randn(self.num_features, self.num_classes) * 0.01
        self.bias = np.zeros(self.num_classes)

    # Methods for Training, Validation, and Testing
    def training(self, X_train, y_train, learning_rate):
        """
        Trains the Multinomial model for one full epoch.
        """
        # 1. Compute predictions (Forward pass handles weights + bias + softmax)
        probs = self.forward(X_train)

        # 2. Compute accuracy
        training_accuracy = self.compute_accuracy(probs, y_train)

        # 3. Compute loss
        training_loss = self.compute_loss(probs, y_train)

        # 4. Compute gradients
        weight_gradients, bias_gradients = self.compute_gradients(X_train, y_train, probs)

        # 5. Update weights using gradient descent
        self.update_weights(weight_gradients, bias_gradients, learning_rate)

        # 6. Return loss and accuracy
        return training_loss, training_accuracy

    def validation(self, X_val, y_val):
        """
        Validates the model and computes accuracy & loss.
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
        Tests the model on test data and computes accuracy.
        """
        # 1. Compute predictions
        probs = self.forward(X_test)

        # 2. Compute accuracy
        testing_accuracy = self.compute_accuracy(probs, y_test)

        # 3. Compute loss
        testing_loss = self.compute_loss(probs, y_test)

        # 4. Return loss and accuracy
        return testing_loss, testing_accuracy
