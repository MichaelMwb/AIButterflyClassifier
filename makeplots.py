import matplotlib.pyplot as plt
import csv

epoch = []
loss = []
training_acc = []
validation_acc = []
with open("stats.txt","r") as f:
    reader = csv.reader(f)
    # Skip the header row: epoch,mean_loss,training_accuracy,validation_accuracy
    next(reader)
    for row in reader:
        if len(row) < 4:
            continue

        epoch.append(int(row[0]))
        loss.append(float(row[1]))
        training_acc.append(float(row[2]))
        validation_acc.append(float(row[3]))

# Plot it
fig, axs = plt.subplots(1, 2, figsize=(8, 6), dpi=144)

axs[0].plot(epoch, loss)
axs[0].set_title("Loss")
axs[0].set_xlabel("Epoch")
axs[0].set_ylabel("Mean Cross-Entropy Loss")

axs[1].set_ylim((0,1.0))
axs[1].plot(epoch, training_acc, color="red", label="Training")
axs[1].plot(epoch, validation_acc, color="blue", label="Validation")
axs[1].set_title("Accuracy")
axs[1].set_xlabel("Epoch")
axs[1].legend()

fig.savefig("learning.png")
fig.show()
