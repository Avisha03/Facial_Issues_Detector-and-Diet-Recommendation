# Facial_Issues_Detector-and-Diet-Recommendation

## 📌 Project Overview
**Facial Issue Detector** is a deep learning-based project that identifies facial skin issues using **Convolutional Neural Networks (CNNs)** and provides **dietary recommendations** based on detected conditions. The model is trained on a dataset of five common skin issues and uses **real-time webcam detection** to classify them accurately.

Once an issue is detected, the system suggests **personalized dietary recommendations** to improve skin health. It also **sends an email report** containing:
- Detected skin issue
- Confidence score
- Recommended diet plan

## 🏆 Problem Statement & Solution
### **🔴 Problem:**
Millions of people suffer from skin conditions but lack proper awareness of their causes and treatments. Dermatologist appointments can be expensive, and self-diagnosis is often inaccurate.

### **✅ Solution:**
- Developed a **CNN-based model** trained on **five facial issues** to detect conditions accurately.
- Integrated a **real-time detection system** using **OpenCV** to capture facial images.
- Implemented a **diet recommendation system** based on detected skin issues.

## 📂 Dataset Details
- **Total Images:** 1,296 (1,197 train + 99 test)
- **Classes:**
  - Acne
  - Eksim (Eczema)
  - Herpes
  - Panu (Tinea Versicolor)
  - Rosacea
- **Format:** Images + CSV file (one-hot encoded labels)

-## 🛠️ Model Training & Improvements

### **🚀 Optimizations & Final Model**
- Switched to **MobileNetV2 (Transfer Learning)**.
- Applied **data augmentation** (flipping, rotation, brightness adjustments).
- Fine-tuned **last two layers** of MobileNetV2.
- Used **Adam optimizer** with **learning rate 1e-4**.
- Achieved **test accuracy: 82.83%**.

## 📷 Real-Time Detection Implementation
### **🔴 Features:**
✔ **Opens webcam with an oval box for face detection**
✔ **Captures image automatically when the face is inside the oval**
✔ **Uses CNN model to classify the skin issue**
✔ **Displays detected issue with confidence score**

### **👨‍💻 How It Works:**
1. **Load Trained Model:** `final_facial_issue_detector.keras`
2. **Open Webcam:** Display oval box overlay using OpenCV.
3. **Face Detection Logic:**
   - If face **outside the oval** → Show **black oval**.
   - If face **inside the oval** → Change oval color to **red**.
4. **Capture Image & Predict Issue**
5. **Display Results & Send Email Report**


## 🚀 Technologies & Tools Used
| **Tool/Library** | **Purpose** |
|----------------|------------|
| TensorFlow / Keras | Deep learning model training |
| OpenCV | Real-time face detection & webcam integration |
| NumPy / Pandas | Data processing |
| Matplotlib / Seaborn | Data visualization |
| Streamlit 

## 🛠 Challenges Faced & Solutions

### **1️⃣ Low Model Accuracy**
❌ **Problem:** The initial CNN model had poor generalization (~65% test accuracy).  
✅ **Solution:** Switched to **MobileNetV2**, added **data augmentation**, and fine-tuned **final layers**.

### **2️⃣ Unstable Webcam Face Detection**
❌ **Problem:** Face detection was inconsistent due to lighting variations.
✅ **Solution:** Used OpenCV’s **face alignment** & adaptive **brightness normalization**.

### **3️⃣ Slow Prediction Time**
❌ **Problem:** Loading large models caused lag in real-time detection.
✅ **Solution:** Converted model to **TensorFlow Lite (TF-Lite)** for faster inference.

## 🎯 Future Improvements
✅ **Enhance Accuracy with more training data**  
✅ **Add more skin conditions for better classification**  
✅ **Deploy model on cloud (AWS/GCP) for real-time API access**  
✅ **Build a Mobile App using Flutter/TensorFlow Lite**

### 🎉 Thank you for exploring this project! 🚀 Feel free to contribute or raise issues! 🤝

