import numpy as np
import matplotlib.pyplot as plt


class PlotGeneration:

    def display_test_accuracies(self, base_test_acc, mlr_test_acc, mlp_test_acc):
        """ Generates a bar plot of all the test accuracies from the different models."""

        accuracies = [base_test_acc * 100, mlr_test_acc * 100, mlp_test_acc * 100]
        labels = ['Baseline Model', 'MLR', 'MLP']

        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(accuracies)), accuracies, tick_label=labels, color=['skyblue', 'lightgreen', 'salmon'])

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')

        plt.xlabel('Model Type', labelpad=15)
        plt.ylabel('Test Accuracy (%)')
        plt.title('Test Accuracies Across Different Models')
        plt.ylim(0, 105)
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.show()

    def display_learning_curves(self, train_acc_history, val_acc, model, model_type):
        """ Generates a Line Plot of the training and validation accuracy over epochs for a model."""

        plt.figure(figsize=(12, 6))
        plt.plot(train_acc_history, label='Training Accuracy', color='blue', marker='o')
        plt.axhline(val_acc, label='Validation Accuracy', color='orange', marker='o')

        if model_type == "MLR":
            plt.title(f'Learning Curves for MLR Model (LR={model["learning_rate"]}, Epochs={model["epochs"]}, Batch Size={model["batch_size"]})')
        else:
            plt.title(f'Learning Curves for MLP Model (Hidden={model["hidden_size"]}, LR={model["learning_rate"]}, Epochs={model["epochs"]}, Batch Size={model["batch_size"]})')

        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1.05)
        plt.xticks(range(0, len(train_acc_history), max(1, len(train_acc_history)//10)))
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def display_loss_curves(self, train_loss_history, model, model_type):
        """ Generates a Line Plot of the training and validation loss over epochs for a model."""

        plt.figure(figsize=(12, 6))
        plt.plot(train_loss_history, label='Training Loss', color='red', marker='o')
        plt.axhline(model['validation_loss'], label='Validation Loss', color='green', marker='o')

        if model_type == "MLR":
            plt.title(f'Loss Curves for MLR Model (LR={model["learning_rate"]}, Epochs={model["epochs"]}, Batch Size={model["batch_size"]})')
        else:
            plt.title(f'Loss Curves for MLP Model (Hidden={model["hidden_size"]}, LR={model["learning_rate"]}, Epochs={model["epochs"]}, Batch Size={model["batch_size"]})')

        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.ylim(0, max(max(train_loss_history), model['validation_loss']) * 1.1)
        plt.xticks(range(0, len(train_loss_history), max(1, len(train_loss_history)//10)))
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrices(self, cm_a, cm_b, class_names, label_a, label_b):
        """
        Displays two confusion matrices side by side for direct visual comparison.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        for ax, cm, label in zip(axes, [cm_a, cm_b], [label_a, label_b]):
            im = ax.imshow(cm, cmap='Blues')
            ax.set_title(f"{label} - Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_xticks(range(len(class_names)))
            ax.set_yticks(range(len(class_names)))
            ax.set_xticklabels(class_names, rotation=45, ha='right')
            ax.set_yticklabels(class_names)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, cm[i, j], ha='center', va='center',
                            color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.show()

    def compare_metrics_bar(self, metrics_df):
        """
        Grouped bar chart comparing macro-averaged metrics between models.
        Expects a DataFrame indexed by model name with metric columns.
        """
        metrics_to_plot = [
            'Accuracy', 'Balanced Accuracy', 'Precision (macro)',
            'Recall (macro)', 'F1-score (macro)', 'PR-AUC (macro)'
        ]
        x = np.arange(len(metrics_to_plot))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        for i, (model_name, row) in enumerate(metrics_df.iterrows()):
            values = [row[m] for m in metrics_to_plot]
            bars = ax.bar(x + (i - 0.5) * width, values, width, label=model_name)
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.2f}", ha='center', va='bottom', fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(metrics_to_plot, rotation=20, ha='right')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1.05)
        ax.set_title('Logistic Regression vs MLP - Test Set Metric Comparison')
        ax.legend()
        plt.tight_layout()
        plt.show()
