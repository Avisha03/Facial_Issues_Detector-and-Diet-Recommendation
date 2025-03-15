import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import base64
import tempfile

# Load the trained model
MODEL_PATH = "final_facial_issue_detector.keras"
model = load_model(MODEL_PATH)

# Define class labels
CLASS_LABELS = ['Acne', 'Eksim', 'Herpes', 'Panu', 'Rosacea']

# Diet recommendations based on skin condition
DIET_RECOMMENDATIONS =  {
    "Acne": """
    🥗 **Best Foods for Acne:**  
    - 🥑 Avocados (Rich in Vitamin E)  
    - 🥕 Carrots (Vitamin A for skin repair)  
    - 🥒 Cucumber (Hydration & Detox)  
    - 🍓 Berries (Antioxidants to reduce inflammation)  

    🚫 **Avoid:** Dairy, Sugar, Fried Foods  
    """,

    "Eksim": """
    🥗 **Best Foods for Eksim (Eczema):**  
    - 🥜 Walnuts (Omega-3 for skin healing)  
    - 🥑 Avocado (Reduces skin dryness)  
    - 🥦 Broccoli (Vitamin C for skin protection)  
    - 🫐 Blueberries (Antioxidants for skin repair)  

    🚫 **Avoid:** Processed foods, Gluten, Spicy foods  
    """,

    "Herpes": """
    🥗 **Best Foods for Herpes:**  
    - 🐟 Salmon (Rich in Omega-3)  
    - 🍌 Bananas (Boosts immunity)  
    - 🍯 Honey (Natural Antiviral properties)  
    - 🥗 Leafy Greens (Rich in Zinc & Vitamins)  

    🚫 **Avoid:** Nuts, Chocolate, Alcohol  
    """,

    "Panu": """
    🥗 **Best Foods for Panu (Tinea Versicolor):**  
    - 🍋 Citrus Fruits (Vitamin C for skin healing)  
    - 🥑 Avocado (Hydration & Repair)  
    - 🥜 Almonds (Boosts skin immunity)  
    - 🥒 Cucumbers (Keeps skin hydrated)  

    🚫 **Avoid:** Excessive Sugary Foods  
    """,

    "Rosacea": """
    🥗 **Best Foods for Rosacea:**  
    - 🍅 Tomatoes (Vitamin C for skin repair)  
    - 🥦 Broccoli (Anti-inflammatory)  
    - 🥥 Coconut Water (Hydration)  
    - 🥕 Carrots (Beta-Carotene for glowing skin)  

    🚫 **Avoid:** Alcohol, Spicy Foods, Caffeine  
    """
}

# Function to set background image
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    page_bg = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{encoded_string}");
        background-size: cover;
    }}
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)

# Streamlit UI
def main():
    st.title("Facial Skin Issue Detector & Diet Recommendation")
    
    # Set background image
    set_background("BG.avif")
    
    # User inputs
    name = st.text_input("Name")
    age = st.number_input("Age", min_value=1, max_value=100, step=1)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    skin_type = st.selectbox("Skin Type", ["Oily", "Dry", "Combination"])
    pregnancy_status = st.selectbox("Pregnancy/Lactation Status", ["Not Applicable", "Pregnant", "Lactating"])
    
    # Webcam Capture
    st.write("### Capture Image for Skin Issue Detection")
    camera = st.camera_input("Take a Picture")
    
    if camera:
        # Convert image for model prediction
        image = Image.open(camera).convert('RGB')
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0  # Normalize
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        
        # Prediction
        predictions = model.predict(image_array)[0]
        predicted_class = CLASS_LABELS[np.argmax(predictions)]
        confidence = np.max(predictions) * 100
        
        # Display result
        st.success(f"**Detected Skin Issue:** {predicted_class} ({confidence:.2f}%)")
        
        # Show diet recommendation
        st.info(f"**Diet Recommendation:** {DIET_RECOMMENDATIONS[predicted_class]}")

if __name__ == "__main__":
    main()
