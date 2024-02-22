from fastai.vision.all import *
import streamlit as st
import pathlib
import plotly.express as px
import platform

plt = platform.system()
if plt == 'Linux' : pathlib.WindowsPath = pathlib.PosixPath

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
st.title("Transportation Classification Model by Mirsaid")
file = st.file_uploader('Upload Picture', type=['png','jpeg', 'gif', 'svg'])

# PIL convert
if file is not None:
    # PIL convert
    st.image(file)
    img  = PILImage.create(file)
    
    # model
    model = load_learner('transport_model.pkl')
    pred, pred_id, probs = model.predict(img)
    
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
