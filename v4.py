import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import torch
from torchvision.transforms import ToTensor
from option import args
from models import InpaintGenerator
import cv2

def postprocess(image):
    image = torch.clamp(image, -1.0, 1.0)
    image = (image + 1) / 2.0 * 255.0
    image = image.permute(1, 2, 0)
    image = image.cpu().numpy().astype(np.uint8)
    return image


# Title
st.title("Save Original Image with Mask and Binary Mask")

# Upload an image
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    # Load the image
    image = Image.open(uploaded_file)
    original_width, original_height = image.size  # Get the original dimensions
    image = image.convert("RGBA")  # Ensure RGBA mode

    # Scale the original image to fit into 512x512 for display purposes
    display_size = (512, 512)
    image.thumbnail(display_size)  # Preserve aspect ratio for display

    # Set up the canvas with the fixed size of 512x512 for display
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",  # White ink
        stroke_width=10,
        stroke_color="white",
        background_image=image,
        height=512,
        width=512,
        drawing_mode="freedraw",
        key="canvas",
    )

    if st.button("Save Original Image with Mask and Binary Mask"):
        if canvas_result.image_data is not None:
            # Get the drawn canvas as RGBA
            drawn_mask_rgba = canvas_result.image_data.astype(np.uint8)

            # Create Binary Mask (Black and White)
            alpha_channel = drawn_mask_rgba[:, :, 3]  # Extract alpha channel
            binary_mask = np.where(alpha_channel > 0, 255, 0).astype(np.uint8)  # Binary mask
            binary_mask_image = Image.fromarray(binary_mask)  # Convert to PIL Image

            # Save the binary mask as a single-channel grayscale image
            binary_mask_image = binary_mask_image.convert("L")  # Convert to grayscale mode
            binary_mask_image.save("binary_mask_single_channel.png")
          
            # Convert the drawn mask to a PIL image and resize it to original image size
            drawn_mask_image = Image.fromarray(drawn_mask_rgba)
            drawn_mask_resized = drawn_mask_image.resize((original_width, original_height), Image.Resampling.LANCZOS)

            # Convert the original image back to an array (to apply the mask)
            original_array = np.array(image.resize((original_width, original_height)))  # Resize back to original
            mask_alpha = np.array(drawn_mask_resized)[:, :, 3] / 255.0  # Normalize alpha channel to [0, 1]

            # Apply the mask to the original image
            for c in range(3):  # Apply mask to R, G, B channels
                original_array[:, :, c] = (
                    original_array[:, :, c] * (1 - mask_alpha)
                    + 255 * mask_alpha  # White color (255)
                )

            # Convert back to an image and save
            masked_image = Image.fromarray(original_array)
            masked_image.save("original_with_mask.png")

            # Display saved images
            # st.image(masked_image, caption="Original Image with Mask", use_column_width=True)
            # st.image(binary_mask_image, caption="Binary Mask", use_column_width=True)
            # load images

            # Model and version
            model = InpaintGenerator(args)
            model.load_state_dict(torch.load(args.pre_train, map_location="cpu"))
            model.eval()

            filename = r"E:\aotgan-streamlit\original_with_mask.png"
            orig_img = cv2.resize(cv2.imread(filename, cv2.IMREAD_COLOR), (512, 512))
            img_tensor = (ToTensor()(orig_img) * 2.0 - 1.0).unsqueeze(0)

            filename = r"E:\aotgan-streamlit\binary_mask_single_channel.png"
            

            mask = cv2.resize(cv2.imread(filename, cv2.IMREAD_GRAYSCALE), (512, 512))

            

            with torch.no_grad():
                mask_tensor = (ToTensor()(mask)).unsqueeze(0)
                print('Shape of Mask : ',mask_tensor.shape)
                masked_tensor = (img_tensor * (1 - mask_tensor).float()) + mask_tensor
                pred_tensor = model(masked_tensor, mask_tensor)
                
                comp_tensor = pred_tensor * mask_tensor + img_tensor * (1 - mask_tensor)

                pred_np = postprocess(pred_tensor[0])
                masked_np = postprocess(masked_tensor[0])
                comp_np = postprocess(comp_tensor[0])
                # Convert the NumPy array to a PIL Image and display it
                comp_image = Image.fromarray(comp_np)
                # If comp_image is a PIL Image, convert it to a NumPy array
                if isinstance(comp_image, Image.Image):
                    comp_image = np.array(comp_image)

                # Resize the image to 512x512
                comp_image_resized = cv2.resize(comp_image, (512, 512))

                # Display the resized image with a caption
                st.image(comp_image_resized, caption="Completed Image", use_column_width=False)
