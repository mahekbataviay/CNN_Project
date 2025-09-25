import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

# --- Configuration ---
st.set_page_config(
    page_title="Digit Recognizer",
    page_icon="✍️",
    layout="wide"
)

# --- Model Loading ---

# Use st.cache_resource to load the model only once, improving performance
@st.cache_resource
def load_mnist_model():
    """Loads the pre-trained MNIST digit recognition model."""
    try:
        # Load the model you trained and saved with train.py
        model = load_model('mnist_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Please ensure the 'mnist_model.h5' file is in the same directory as this script.")
        return None

model = load_mnist_model()

# --- Image Preprocessing ---

def preprocess_image(image_data):
    """
    Preprocesses the canvas image to fit the model's input requirements.
    The model was trained on 28x28 grayscale images.
    """
    # The canvas returns an RGBA image, so we convert it to grayscale ('L' mode)
    img = Image.fromarray(image_data.astype('uint8'), 'RGBA').convert('L')
    
    # Resize the image to the 28x28 pixels the model expects
    img = img.resize((28, 28))
    
    # Convert the image to a numpy array
    img_array = np.array(img)
    
    # Reshape the array for the model: (1, 28, 28, 1) -> 1 sample, 28x28 pixels, 1 channel (grayscale)
    img_array = img_array.reshape(1, 28, 28, 1)
    
    # Normalize pixel values from [0, 255] to [0, 1], just like the training data
    img_array = img_array.astype('float32') / 255.0
    
    return img_array

# --- Streamlit UI ---

st.title("✍️ Handwritten Digit Recognizer")
st.write(
    "Draw a single digit (from 0 to 9) in the black box below. "
    "For the best results, try to draw the digit large and centered."
)

# Create two columns for the layout
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Drawing Canvas")
    # Create the drawable canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Not used in freedraw mode
        stroke_width=20, # Thickness of the drawing pen
        stroke_color="#FFFFFF", # Color of the pen (white)
        background_color="#000000", # Background of the canvas (black)
        height=400,
        width=400,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    st.header("Prediction")
    # Only show the predict button if the model is loaded and something is drawn
    if model is not None and canvas_result.image_data is not None:
        if st.button("Predict"):
            # Get the image data from the canvas
            image_data = canvas_result.image_data
            
            # Check if the canvas is not empty
            if np.sum(image_data) > 0:
                # Preprocess the image to be model-ready
                processed_image = preprocess_image(image_data)
                
                # Use the model to predict the digit
                prediction = model.predict(processed_image)
                predicted_digit = np.argmax(prediction)
                confidence = np.max(prediction) * 100
                
                # Display the prediction results
                st.success(f"## Predicted Digit: **{predicted_digit}**")
                st.metric(label="Confidence", value=f"{confidence:.2f}%")
                
                # Show the full probability distribution in an expandable section
                with st.expander("See probability distribution"):
                    st.bar_chart(prediction.flatten())
            else:
                st.warning("The canvas is empty. Please draw a digit first!")
    else:
        st.info("Draw a digit on the canvas and click the 'Predict' button.")

st.sidebar.header("About")
st.sidebar.info(
    "This web app uses a Convolutional Neural Network (CNN), "
    "built with TensorFlow/Keras, to recognize handwritten digits drawn on the canvas. "
    "The model was trained on the classic MNIST dataset."
)

