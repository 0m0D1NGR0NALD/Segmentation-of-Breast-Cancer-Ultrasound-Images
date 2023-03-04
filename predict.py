import matplotlib.pyplot as plt

def prepare_plot(orig_img,orig_mask,pred_mask):
    # Initialize our figure
    figure,ax = plt.subplots(nrows=1,ncols=3,figsize=(10,10)):
    # Plot the original image, original mask, predicted mask
    ax[0].imshow(orig_mask)
    ax[1].imshow(orig_mask)
    ax[2].imshow(pred_mask)
    # Set the titles of the subplots
    ax[0].set_title("Image")
    ax[1].set_title("Original Mask")
    ax[2].set_title("Predicted Mask")
    # Set the layout of the figure
    figure.tight_layout()
    # Display figure
    figure.show()