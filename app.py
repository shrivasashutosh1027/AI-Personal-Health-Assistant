# =============================
# app.py - 3-in-1 Health Diagnostic AI
# =============================

import streamlit as st
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -----------------------------
# DEVICE
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================
# 1️⃣ SYMPTOMS → DISEASE MODULE
# =============================
# Load CSVs
df = pd.read_csv("dataset.csv")
desc = pd.read_csv("symptom_Description.csv")
prec = pd.read_csv("symptom_precaution.csv")

# Strip column spaces
df.columns = df.columns.str.strip()
desc.columns = desc.columns.str.strip()
prec.columns = prec.columns.str.strip()

# Combine symptom columns
symptom_cols = [c for c in df.columns if "Symptom" in c]
df["All_Symptoms"] = df[symptom_cols].astype(str).apply(
    lambda row: " ".join([s.replace(" ","_") for s in row if s != "nan"]), axis=1
)

X = df["All_Symptoms"]
y = df["Disease"]

# Train/Test split + TF-IDF
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Logistic Regression Model
lr_model = LogisticRegression(max_iter=500)
lr_model.fit(X_train_tfidf, y_train)

# Prediction Function
def predict_symptoms(symptoms):
    input_text = " ".join([s.replace(" ","_") for s in symptoms])
    input_tfidf = vectorizer.transform([input_text])
    disease = lr_model.predict(input_tfidf)[0]

    # Description
    description = desc[desc['Disease']==disease]["Description"].values
    description = description[0] if len(description)>0 else "No description available."
    
    # Precautions
    p = prec[prec["Disease"]==disease]
    precautions = p.iloc[0,1:].dropna().tolist() if not p.empty else ["No precaution listed"]

    return disease, description, precautions

# =============================
# 2️⃣ CHEST X-RAY → PNEUMONIA MODULE
# =============================
xray_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])

xray_model = models.resnet18(pretrained=True)
xray_model.fc = nn.Linear(xray_model.fc.in_features, 2)
xray_model.load_state_dict(torch.load("pneumonia_model.pth", map_location=device))
xray_model = xray_model.to(device)
xray_model.eval()

def predict_xray(img):
    img = img.convert("RGB")
    img_tensor = xray_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = xray_model(img_tensor)
        _, predicted = torch.max(outputs, 1)
    classes = ["NORMAL", "PNEUMONIA"]
    return classes[predicted.item()]

# =============================
# 3️⃣ SKIN CANCER → HAM10000 MODULE
# =============================
skin_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])

# Class mapping
skin_classes = ['bkl', 'nv', 'df', 'mel', 'vasc', 'bcc', 'akiec']
skin_class_names = {
    'bkl':"Benign keratosis-like lesions",
    'nv':"Melanocytic nevi",
    'df':"Dermatofibroma",
    'mel':"Melanoma",
    'vasc':"Vascular lesions",
    'bcc':"Basal cell carcinoma",
    'akiec':"Actinic keratoses / Intraepithelial carcinoma"
}

skin_model = models.resnet18(pretrained=True)
skin_model.fc = nn.Linear(skin_model.fc.in_features, len(skin_classes))
skin_model.load_state_dict(torch.load("skin_cancer_model.pth", map_location=device))
skin_model = skin_model.to(device)
skin_model.eval()

def predict_skin(img):
    img = img.convert("RGB")
    img_tensor = skin_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = skin_model(img_tensor)
        _, predicted = torch.max(outputs,1)
    code = skin_classes[predicted.item()]
    return skin_class_names[code]

# =============================
# STREAMLIT GUI
# =============================
# -----------------------------
# INTRODUCTION PAGE
# -----------------------------
st.set_page_config(page_title="Health AI Assistant", layout="wide")

# Sidebar for navigation
module = st.sidebar.radio("Navigation", ["Introduction", "Symptoms Checker", "Chest X-ray", "Skin Cancer"])

st.markdown(
    """
    <style>
    .reportview-container {
        background: linear-gradient(to right, #e0f7fa, #81d4fa);
        font-family: 'Helvetica', sans-serif;
    }
    .sidebar .sidebar-content {
        background-color: #b2dfdb;
        padding: 10px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #00796b;
        color: white;
        border-radius: 10px;
        height: 50px;
        width: 100%;
        font-size:16px;
    }
    </style>
    """, unsafe_allow_html=True
)

if module == "Introduction":
    st.title("🧠 Health AI Assistant")
    st.subheader("Welcome!")
    st.markdown("""
        This AI-powered assistant helps you:
        - Predict diseases based on **Symptoms**
        - Detect **Pneumonia** from **Chest X-ray images**
        - Detect **Skin Cancer** from **Skin Lesion images**
        
        **Instructions:**
        1. Use the sidebar to select a module.
        2. Enter symptoms or upload images as required.
        3. Click the Predict button to get results.
        
        Stay healthy and explore AI-driven diagnostics!
    """)
    st.image(
        "ChatGPT Image Aug 28, 2025, 02_40_38 AM.png", 
        caption="Health AI Assistant Overview",
        use_container_width=True,  # makes it fit the page width
        clamp=True,             # ensures image stays within container
    )  #Adding an intro image


# -----------------------------
# Symptoms Checker Tab
# -----------------------------
if module=="Symptoms Checker":
    st.header("Symptoms → Disease Prediction")
    
    # Dropdown of all diseases from the dataset
    all_diseases = df['Disease'].unique().tolist()
    selected_disease = st.selectbox("Select a Disease", all_diseases)
    
    if st.button("Show Details"):
        # Get description
        description = desc[desc['Disease']==selected_disease]["Description"].values
        description = description[0] if len(description)>0 else "No description available."
        
        # Get precautions
        p = prec[prec["Disease"]==selected_disease]
        precautions = p.iloc[0,1:].dropna().tolist() if not p.empty else ["No precaution listed"]
        
        st.success(f"Disease: {selected_disease}")
        st.info(f"Description: {description}")
        st.write("Precautions:")
        for i, pr in enumerate(precautions,1):
            st.write(f"{i}. {pr}")


# -----------------------------
# Chest X-ray Tab
# -----------------------------
elif module=="Chest X-ray":
    st.header("Chest X-ray → Pneumonia Prediction")
    uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg","png","jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)
        if st.button("Predict Pneumonia"):
            result = predict_xray(img)
            st.success(f"Prediction: {result}")

# -----------------------------
# Skin Cancer Tab
# -----------------------------
elif module=="Skin Cancer":
    st.header("Skin Lesion → Cancer Type Prediction")
    uploaded_file = st.file_uploader("Upload Skin Lesion Image", type=["jpg","png","jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)
        if st.button("Predict Skin Cancer"):
            result = predict_skin(img)
            st.success(f"Prediction: {result}")
