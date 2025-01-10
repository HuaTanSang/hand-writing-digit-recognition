import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np
import model_directory as dir 


# Load model
# @st.cache(allow_output_mutation=True)
def load_model(model_path):
    model = torch.load(model_path)
    return model

# Preprocess input image
def preprocess_image(image):
    # Convert to grayscale, resize to 28x28, and invert colors
    image = ImageOps.grayscale(image)
    image = image.resize((28, 28))
    # Normalize pixel values
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    tensor_image = transform(image)
    tensor_image = tensor_image.unsqueeze(0)  # Add batch dimension
    return tensor_image

# Display top predictions
def get_predictions(model, image):
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1).squeeze()
        top5_prob, top5_classes = torch.topk(probabilities, 5)
    return top5_classes.numpy(), top5_prob.numpy()

# Streamlit App
st.title("MNIST Digit Recognition Demo")

# Upload model
model_path = st.file_uploader("Upload your .pth model file", type=["pth"])
if model_path:
    model = load_model(model_path)

    # Drawing area
    st.subheader("Draw a digit below:")
    canvas_result = st.canvas(
        fill_color="white",  # Background color
        stroke_width=10,
        stroke_color="black",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas"
    )

    if canvas_result.image_data is not None:
        drawn_image = canvas_result.image_data
        # Convert to PIL image
        pil_image = Image.fromarray((255 - drawn_image[:, :, 3]).astype("uint8"))  # Use alpha channel for drawing

        # Preprocess and predict
        tensor_image = preprocess_image(pil_image)
        top5_classes, top5_probs = get_predictions(model, tensor_image)

        # Display predictions
        st.subheader("Top 5 Predictions:")
        for i, (cls, prob) in enumerate(zip(top5_classes, top5_probs)):
            st.write(f"{i + 1}: Digit {cls} with probability {prob:.4f}")
