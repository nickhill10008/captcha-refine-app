import streamlit as st
from generate_captchas import generate_one, random_text
from refine_loop import refine_loop
from PIL import Image
import os

st.title("🔐 CAPTCHA Refinement Project")

# Generate a random captcha
if st.button("Generate CAPTCHA"):
    text = random_text()
    img = generate_one(text)
    st.image(img, caption=f"Generated CAPTCHA: {text}")

# Run refinement loop
if st.button("Run Refinement (10 iters)"):
    st.write("Running refine loop... (check logs below)")
    refine_loop(iters=10, batch=4)
    st.success("Refinement finished!")

# Upload captcha and test
uploaded = st.file_uploader("Upload a CAPTCHA image to test", type=["png", "jpg"])
if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded CAPTCHA")
    st.write("📊 Breaker model evaluation coming soon...")
