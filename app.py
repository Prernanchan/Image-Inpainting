import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the trained autoencoder model
autoencoder = tf.keras.models.load_model("model.h5")

# Constants for the blackening area
left = 60
right = 76
up = 78
down = 62

def inpaint_image(image):
    # Resize the image to 128x128
    image = image.resize((128, 128))

    # Convert the image to numpy array
    image_array = np.array(image)

    # Create a copy of the image
    black_image = image_array.copy()

    # Blacken out the specified region
    black_image[left:right, down:up, :] = 0

    # Perform inpainting using the trained autoencoder model
    prediction = autoencoder.predict(black_image.reshape(-1, 128, 128, 3))

    # Get the inpainted patch with the same dimensions as the blackened region
    inpainted_patch = prediction[0, left:right, down:up, :]

    # Replace the blackened region with the inpainted patch
    inpainted_image = black_image.copy()
    inpainted_image[left:right, down:up, :] = inpainted_patch

    return Image.fromarray(inpainted_image.astype(np.uint8))

def main():
    st.title("Image Inpainting Web App")
    st.write("Select an image and specify the area to be blackened out.")

    # Upload image file
    image_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if image_file is not None:
        # Load the image
        image = Image.open(image_file)

        # Display the original image
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

        # Create a copy of the image to display the blackened area
        blackened_image = image.copy()

        # Blacken out the specified region in the blackened image
        blackened_image_array = np.array(blackened_image)
        blackened_image_array[left:right, down:up, :] = 0

        # Display the blackened image
        st.subheader("Image with Blackened Area")
        st.image(blackened_image_array, use_column_width=True)

        # Inpaint the blackened area and display the result
        inpainted_image = inpaint_image(image)
        st.subheader("Inpainted Image")
        st.image(inpainted_image, use_column_width=True)

# Run the web app
if __name__ == "__main__":
    main()
