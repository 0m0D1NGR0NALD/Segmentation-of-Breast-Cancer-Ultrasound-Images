import matplotlib.pyplot as plt
from train import history

history = {"train_loss":[1,2,3,4],"val_loss":[1,2,3,4]}

# Plot the training loss
plt.style.use("ggplot")
plt.figure()
plt.plot(history["train_loss"],label="train_loss")
plt.plot(history["val_loss"],label="val_loss")
plt.title("Training Loss on Dataset")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(loo="lower left")
plt.savefig("Loss_Curves.png")
