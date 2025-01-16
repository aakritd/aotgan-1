import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import torch
from torchvision.transforms import ToTensor
from models import InpaintGenerator
import cv2
from option import args

# Define a postprocessing function
def postprocess(image):
    image = torch.clamp(image, -1.0, 1.0)
    image = (image + 1) / 2.0 * 255.0
    image = image.permute(1, 2, 0)
    image = image.cpu().numpy().astype(np.uint8)
    return image

# Streamlit custom styling
st.markdown("""
    <style>
        body {
            background-color: #121212;
            color: #E0E0E0;
            font-family: 'Arial', sans-serif;
        }
        .stButton button {
            background-color: #FF6F61 !important;
            color: white !important;
            font-weight: bold;
            border-radius: 5px;
            padding: 10px;
        }
        .stButton button:hover {
            background-color: #D45D54 !important;
        }
        .container {
            background-color: #1F1F1F;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        }
        .title {
            color: #FF6F61;
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            margin-bottom: 20px;
        }
        .result-image {
            margin-top: 20px;
            display: flex;
            justify-content: center;
        }
        .result-image img {
            width: 80%;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
        }
        .file-uploader {
            border: 2px solid #FF6F61;
            border-radius: 8px;
            padding: 10px;
            background-color: #1F1F1F;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit Title
st.markdown('<div class="title">Image Inpainting with AOT GAN Model</div>', unsafe_allow_html=True)

# Container for the app content
st.markdown('<div class="container">', unsafe_allow_html=True)

# Upload Image
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], key="upload", label_visibility="collapsed")
if uploaded_file:
    # Load and preprocess the image
    image = Image.open(uploaded_file).convert("RGB")
    original_width, original_height = image.size  # Get original dimensions
    display_size = (512, 512)
    image_resized = image.resize(display_size, Image.Resampling.LANCZOS)
    original_array = np.array(image)

    # Display Canvas for drawing mask
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",
        stroke_width=10,
        stroke_color="white",
        background_image=image_resized,
        height=512,
        width=512,
        drawing_mode="freedraw",
        key="canvas",
    )

    if st.button("Perform Inpainting"):
        if canvas_result.image_data is not None:
            # Extract mask from the canvas
            drawn_mask_rgba = canvas_result.image_data.astype(np.uint8)
            alpha_channel = drawn_mask_rgba[:, :, 3]  # Extract alpha channel
            binary_mask = np.where(alpha_channel > 0, 255, 0).astype(np.uint8)

            # Save binary mask (optional, for debugging)
            binary_mask_image = Image.fromarray(binary_mask)
            binary_mask_image.save("binary_mask_single_channel.png")

            # Preprocess the image and mask
            original_array_resized = cv2.resize(original_array, display_size)
            img_tensor = (ToTensor()(original_array_resized) * 2.0 - 1.0).unsqueeze(0)

            mask_tensor = torch.tensor(binary_mask).float().unsqueeze(0) / 255.0

            # Load the model
            model = InpaintGenerator(args)
            model.load_state_dict(torch.load(args.pre_train, map_location="cpu"))
            model.eval()

            with torch.no_grad():
                mask_tensor = torch.tensor(binary_mask).float().unsqueeze(0) / 255.0
                mask_tensor = mask_tensor.unsqueeze(1)  # Add a channel dimension

                # Generate masked image
                masked_tensor = img_tensor * (1 - mask_tensor) + mask_tensor
                pred_tensor = model(masked_tensor, mask_tensor)
                comp_tensor = pred_tensor * mask_tensor + img_tensor * (1 - mask_tensor)

                # Postprocess results
                pred_np = postprocess(pred_tensor[0])
                masked_np = postprocess(masked_tensor[0])
                comp_np = postprocess(comp_tensor[0])

                # Display results in a clean, organized layout
                st.markdown('<div class="result-image">', unsafe_allow_html=True)
                st.image(comp_np, caption="Completed Image", use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
