import matplotlib.pyplot as plt


epochs = [1, 2, 3, 4, 5]

train_loss = [2.4795, 0.9795, 0.6154, 0.4392, 0.3448]
val_loss = [1.2543, 0.9285, 0.8508, 0.7945, 0.7834]

train_acc = [0.4327, 0.7297, 0.8355, 0.8800, 0.8993]
val_acc = [0.6697, 0.7411, 0.7518, 0.7603, 0.7766]

train_top_3_acc = [0.6418, 0.9180, 0.9557, 0.9763, 0.9844]
val_top_3_acc = [0.8655, 0.9111, 0.9173, 0.9257, 0.9201]


fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

axes[0].plot(epochs, train_loss, marker="o", label="Train Loss")
axes[0].plot(epochs, val_loss, marker="o", label="Validation Loss")
axes[0].set_title("Training and Validation Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_xticks(epochs)
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].plot(epochs, train_acc, marker="o", label="Train Accuracy")
axes[1].plot(epochs, val_acc, marker="o", label="Validation Accuracy")
axes[1].set_title("Training and Validation Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_xticks(epochs)
axes[1].set_ylim(0.0, 1.0)
axes[1].grid(True, alpha=0.3)
axes[1].legend()

axes[2].plot(epochs, train_top_3_acc, marker="o", label="Train Top-3 Accuracy")
axes[2].plot(epochs, val_top_3_acc, marker="o", label="Validation Top-3 Accuracy")
axes[2].set_title("Training and Validation Top-3 Accuracy")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("Top-3 Accuracy")
axes[2].set_xticks(epochs)
axes[2].set_ylim(0.0, 1.0)
axes[2].grid(True, alpha=0.3)
axes[2].legend()

fig.tight_layout()
fig.savefig("baseline_training_curves.png", dpi=200, bbox_inches="tight")
