import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# Set up the Streamlit app
st.title("Draw on an Image")

# Load an image
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Configure canvas
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Transparent fill with orange color
        stroke_width=10,
        stroke_color="white",
        background_image=image,
        update_streamlit=True,
        height=image.height,
        width=image.width,
        drawing_mode="freedraw",  # Options: "freedraw", "rect", "circle", "transform", etc.
        key="canvas",
    )

    # Save the drawing
    if st.button("Save Drawing"):
        if canvas_result.image_data is not None:
            drawn_image = Image.fromarray((canvas_result.image_data).astype("uint8"))
            drawn_image.save("output_image.png")
            st.success("Drawing saved as output_image.png!")
