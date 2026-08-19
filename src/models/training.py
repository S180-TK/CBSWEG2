import numpy as np
from sklearn.model_selection import StratifiedKFold


def run_kfold_valid(model, X, y, learning_rate, max_epochs, k, batch_size, patience_limit, cols_to_scale, scaler):
    """
    Runs k-fold cross-validation for the current configuration
    """

    # Initialize K-fold object
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    # Trackers for fold
    fold_accuracies = []
    fold_losses = []

    y_labels = np.argmax(y, axis=1) if len(y.shape) > 1 else y

    for fold_num, (train_idx, val_idx) in enumerate(kf.split(X, y_labels), start=1):
        X_fold_train, X_fold_val = X[train_idx], X[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]

        # Scale using a scaler fitted on this fold's training data only
        X_fold_train[:, cols_to_scale] = scaler.fit_transform(X_fold_train[:, cols_to_scale])
        X_fold_val[:, cols_to_scale] = scaler.transform(X_fold_val[:, cols_to_scale])

        # Fresh weights for every fold
        model.reset_weights()

        best_fold_acc = 0
        best_fold_loss = None
        patience_counter = 0

        # Used for batch loop
        num_samples = X_fold_train.shape[0]

        for epoch in range(max_epochs):

            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_shuffled = X_fold_train[indices]
            y_shuffled = y_fold_train[indices]

            # MINI-BATCH LOOP
            for start_idx in range(0, num_samples, batch_size):
                end_idx = start_idx + batch_size

                # Slice the current batch
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                # Update weights using ONLY this batch
                # No need to track training loss and accuracy here
                model.training(X_batch, y_batch, learning_rate)

            val_loss, val_acc = model.validation(X_fold_val, y_fold_val)

            if val_acc > best_fold_acc:
                best_fold_acc = val_acc
                best_fold_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience_limit:
                    break

        fold_accuracies.append(best_fold_acc)
        fold_losses.append(best_fold_loss)
        print(f"  Fold {fold_num}/{k} | Val Acc: {best_fold_acc*100:.2f}% | Val Loss: {best_fold_loss:.4f}")

    return fold_accuracies, fold_losses


def train_and_evaluate_final_model(model, X_train, y_train, X_val, y_val, X_test, y_test, lr, max_epochs, batch_size, patience_limit):
    """
    Trains a model, applies early stopping, saves the best weights,
    and evaluates on the Test Set.
    """
    # Trackers for the Learning Curve plots
    train_acc_history = []
    train_loss_history = []


    # Trackers for the best metrics
    best_val_acc = 0.0
    best_val_loss = 0.0
    best_train_acc = 0.0
    patience_counter = 0
    best_weights = None

    # Used for batch loop
    num_samples = X_train.shape[0]

    for epoch in range(max_epochs):

        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        batch_train_loss = 0.0
        batch_train_acc = 0.0
        num_batches = 0

        # MINI-BATCH LOOP
        for start_idx in range(0, num_samples, batch_size):
            end_idx = start_idx + batch_size

            # Slice the current batch
            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]

            # Update weights using ONLY this batch
            train_loss, train_acc = model.training(X_batch, y_batch, lr)

            # Add the training losses and accuracies, and track number of batches
            batch_train_loss += train_loss
            batch_train_acc += train_acc
            num_batches += 1

        # Get the Average of training loss and accuracies
        epoch_train_acc = batch_train_acc / num_batches
        epoch_train_loss = batch_train_loss / num_batches

        # Get validation loss and accuracy
        val_loss, val_acc = model.validation(X_val, y_val)

        # Add epoch training accuracy and loss to history
        train_acc_history.append(epoch_train_acc)
        train_loss_history.append(epoch_train_loss)

        # Print progress safely
        if epoch % 50 == 0:
            print(f"  Epoch {epoch:4d} | Train Acc: {epoch_train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

        # Early Stopping Logic
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_train_acc = epoch_train_acc
            patience_counter = 0

            # Save the peak weights
            best_weights = model.get_weight_and_bias()
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"  -> Early stopping at epoch {epoch}! Best Val Acc: {best_val_acc*100:.2f}%")
                break

    # Lock in the best weights
    model.set_weights(best_weights)

    # Final Evaluation on the Test Set
    test_loss, test_acc = model.validation(X_test, y_test)

    # Package Results
    results = {
        "learning_rate": lr,
        "epochs": max_epochs,
        "batch_size": batch_size,
        "patience_limit": patience_limit,
        "training_accuracy": best_train_acc,
        "validation_loss": best_val_loss,
        "validation_accuracy": best_val_acc,
        "testing_loss": test_loss,
        "testing_accuracy": test_acc
    }

    # Return results, training accuracy and loss history
    return results, train_acc_history, train_loss_history
