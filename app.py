from fastai.vision.all import *
import streamlit as st
import pathlib
import plotly.express as px
import platform

plt = platform.system()
if plt == 'Linux' : pathlib.WindowsPath = pathlib.PosixPath

st.title("Transportation Classification Model by Mirsaid")
file = st.file_uploader('Upload Picture', type=['png','jpeg', 'gif', 'svg'])

# PIL convert

def load_model():
    model_file_path = "transport_model.pkl"

    # Open the file in binary mode
    with open(model_file_path, 'rb') as model_file:
        # Load the model using torch.load and specify map_location if needed
        model = torch.load(model_file, map_location=torch.device('cpu'))
    
    return model

if file is not None:
    # PIL convert
    st.image(file)
    img  = PILImage.create(file)
    
    # Function to load model


    # Load the model
    model = load_model()

    
    # text
    st.success(f"Prediction:  {pred}")
    st.info(f"Probability: {probs[pred_id]*100:.1f}%")
    
    
    fig = px.bar(y=probs*100, x=model.dls.vocab)
    fig.update_layout(
    yaxis_title="Probability(%)",  # Label for the y-axis
    xaxis_title="Categories"        # Label for the x-axis
    )
    st.plotly_chart(fig)

else:
    st.write("No image uploaded. Please upload image!")

