import streamlit as st
from refine_loop import refine_loop
from PIL import Image

# Set up the page
st.set_page_config(
    page_title="CAPTCHA Refine Demo",
    page_icon="🔐",
    layout="wide"
)

st.title("🔐 CAPTCHA Refinement Demo")

st.markdown("""
This app refines CAPTCHA images using a Machine Learning-based approach.
Upload an image or click "Run Refinement" to see the results.
""")

# Upload CAPTCHA image
uploaded_file = st.file_uploader("Upload CAPTCHA Image", type=["png","jpg","jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Original CAPTCHA", use_column_width=True)

    # Run refinement
    if st.button("Run Refinement"):
        st.write("Refining image...")
        refined_img = refine_loop(img)  # assumes refine_loop returns a PIL image
        st.image(refined_img, caption="Refined CAPTCHA", use_column_width=True)

# Optional: run batch refinement (demo)
if st.button("Run Demo Batch Refinement"):
    st.write("Running batch refinement on sample images...")
    refine_loop(10, batch=2)  # adjust based on your refine_loop parameters
    st.success("Batch refinement completed!")

