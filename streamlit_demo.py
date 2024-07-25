import streamlit as st
from PIL import Image
from inference import TrainStylePredictor
from torchvision import transforms
import numpy as np
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    return TrainStylePredictor()

trainer = load_model()

def style_transfer(input_image, style_description):
    output_image = trainer.test(input_image, style_description)
    return output_image

def main():
    st.title("FastCLIPstyler Demo")

    uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        file_type = uploaded_file.name.split('.')[-1]

        if file_type in ["jpg", "png", "jpeg"]:
            image = Image.open(uploaded_file)
            st.image(image, caption='Content Image', use_column_width=True)

            transform = transforms.Compose([transforms.ToTensor()])
            input_image = transform(np.array(image))[:3, :, :]
            input_image = input_image.unsqueeze(0).to(device)

        else:
            st.error("Unsupported file type")

    style_description = st.text_input("Enter the style description...", "sketch with crayon")

    if st.button("Transfer Style"):
        if uploaded_file is not None:
            if file_type in ["jpg", "png", "jpeg"]:
                output_image = style_transfer(input_image, style_description)[0]
                to_pil = transforms.ToPILImage()
                output_image = to_pil(output_image)
                st.image(output_image, caption='Stylized Image', use_column_width=True)

        else:
            st.warning("Please upload an image in jpg or png format.")

if __name__ == "__main__":
    main()
