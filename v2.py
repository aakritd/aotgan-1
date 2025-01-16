import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np

# Title
st.title("Save Masked Image and Binary Mask")

# Upload an image
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file)

    # Canvas for drawing
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0.5)",  # Semi-transparent white
        stroke_width=5,
        stroke_color="white",
        background_image=image,
        height=image.height,
        width=image.width,
        drawing_mode="freedraw",
        key="canvas",
    )

    # Save the masked image and binary mask
    if st.button("Save Masked Image and Binary Mask"):
        if canvas_result.image_data is not None:
            # Get the drawn mask (RGBA)
            drawn_mask_rgba = canvas_result.image_data.astype(np.uint8)

            # Convert to binary mask (black & white)
            binary_mask = np.where(drawn_mask_rgba[:, :, 3] > 0, 255, 0).astype(np.uint8)
            binary_mask = Image.fromarray(binary_mask)  # Convert to PIL Image
            binary_mask.save("binary_mask.png")
            st.success("Binary mask saved as 'binary_mask.png'!")

            # Apply the mask to the original image
            original_image = np.array(image)
            mask_alpha = drawn_mask_rgba[:, :, 3] / 255.0
            masked_image = (original_image * mask_alpha[:, :, None]).astype(np.uint8)
            masked_image = Image.fromarray(masked_image)
            masked_image.save("masked_image.png")
            st.success("Masked image saved as 'masked_image.png'!")

            st.image(binary_mask, caption="Binary Mask", use_column_width=True)
            st.image(masked_image, caption="Masked Image", use_column_width=True)
