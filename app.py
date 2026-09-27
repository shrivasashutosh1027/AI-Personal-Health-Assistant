import os
from pathlib import Path

import streamlit as st
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI 3-in-1 Health Care System",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_FILE = BASE_DIR / "dataset.csv"
DESCRIPTION_FILE = BASE_DIR / "symptom_Description.csv"
PRECAUTION_FILE = BASE_DIR / "symptom_precaution.csv"

PNEUMONIA_MODEL_FILE = BASE_DIR / "pneumonia_model.pth"
SKIN_MODEL_FILE = BASE_DIR / "skin_cancer_model.pth"

PROJECT_IMAGE = BASE_DIR / "ChatGPT Image Aug 28, 2025, 02_40_38 AM.png"


# ============================================================
# MODEL INFORMATION
# ============================================================

XRAY_MODEL_ACCURACY = "84.29%"
SKIN_MODEL_ACCURACY = "74.47%"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL
       ======================================================== */

    .stApp {
        background-color: #f5f8f6;
        color: #123b37;
    }

    .main .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #123b37 !important;
    }

    p {
        color: #234d48;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {
        background-color: #0b3d39 !important;
        min-width: 280px !important;
        max-width: 280px !important;
    }

    section[data-testid="stSidebar"] > div {
        background-color: #0b3d39 !important;
    }

    section[data-testid="stSidebar"] * {
        color: white;
    }

    section[data-testid="stSidebar"] .stRadio label {
        color: white !important;
        font-weight: 500 !important;
    }

    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
        gap: 8px;
    }

    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label {
        padding: 10px 12px;
        border-radius: 8px;
    }

    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:hover {
        background-color: rgba(255, 255, 255, 0.08);
    }


    /* ========================================================
       SIDEBAR BRAND
       ======================================================== */

    .sidebar-brand {
        padding: 8px 4px 24px 4px;
    }

    .sidebar-kicker {
        color: #7dd3c7;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 2px;
        margin-bottom: 8px;
        text-transform: uppercase;
    }

    .sidebar-title {
        color: white;
        font-size: 25px;
        font-weight: 800;
        line-height: 1.15;
        margin-bottom: 10px;
    }

    .sidebar-copy {
        color: #d7ebe8;
        font-size: 13px;
        line-height: 1.6;
    }

    .sidebar-divider {
        height: 1px;
        background-color: rgba(255, 255, 255, 0.2);
        margin: 12px 0 22px 0;
    }

    .sidebar-disclaimer {
        color: #c8dedb;
        font-size: 11px;
        line-height: 1.6;
        margin-top: 24px;
    }


    /* ========================================================
       NATIVE STREAMLIT SIDEBAR BUTTON
       ======================================================== */

    [data-testid="stSidebarCollapseButton"] {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        z-index: 999999 !important;
    }

    [data-testid="stSidebarCollapseButton"] button {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        align-items: center !important;
        justify-content: center !important;
        background-color: #0b3d39 !important;
        color: white !important;
        border: 1px solid #0b3d39 !important;
        border-radius: 6px !important;
        width: 36px !important;
        height: 36px !important;
        box-shadow: none !important;
    }

    [data-testid="stSidebarCollapseButton"] button:hover {
        background-color: #14534d !important;
        border-color: #14534d !important;
    }

    [data-testid="stSidebarCollapseButton"] button svg {
        color: white !important;
        fill: white !important;
        stroke: white !important;
    }


    /* Collapsed sidebar button */

    [data-testid="stSidebarCollapsedControl"] {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        z-index: 999999 !important;
    }

    [data-testid="stSidebarCollapsedControl"] button {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        align-items: center !important;
        justify-content: center !important;
        background-color: #0b3d39 !important;
        color: white !important;
        border: 1px solid #0b3d39 !important;
        border-radius: 6px !important;
        width: 36px !important;
        height: 36px !important;
        box-shadow: none !important;
    }

    [data-testid="stSidebarCollapsedControl"] button:hover {
        background-color: #14534d !important;
        border-color: #14534d !important;
    }

    [data-testid="stSidebarCollapsedControl"] button svg {
        color: white !important;
        fill: white !important;
        stroke: white !important;
    }


    /* ========================================================
       HOME / OVERVIEW
       ======================================================== */

    .home-kicker {
        color: #087f72;
        font-size: 13px;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .home-title {
        color: #123b37;
        font-size: 50px;
        font-weight: 850;
        line-height: 1.12;
        margin-bottom: 24px;
    }

    .home-description {
        color: #183f3a;
        font-size: 17px;
        line-height: 1.8;
        max-width: 760px;
        margin-bottom: 30px;
    }


    /* ========================================================
       AVAILABLE TOOLS
       ======================================================== */

    .tools-title {
        color: #087f72;
        font-size: 13px;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
        padding-bottom: 12px;
        border-bottom: 3px solid #087f72;
        margin-top: 24px;
        margin-bottom: 0;
    }

    .feature-item {
        padding: 24px 0 22px 0;
        border-bottom: 3px solid #e8b12b;
    }

    .feature-item.gold {
        border-bottom-color: #e8b12b;
    }

    .feature-item.coral {
        border-bottom-color: #df6852;
    }

    .feature-title {
        color: #123b37;
        font-size: 19px;
        font-weight: 750;
        margin-bottom: 10px;
    }

    .feature-copy {
        color: #52746f;
        font-size: 16px;
        line-height: 1.6;
    }


    /* ========================================================
       IMAGE
       ======================================================== */

    .hero-image {
        width: 100%;
        max-height: 500px;
        object-fit: cover;
        border-radius: 10px;
    }


    /* ========================================================
       NOTE BOX
       ======================================================== */

    .note-box {
        border-left: 3px solid #e8b12b;
        padding: 12px 20px;
        margin-top: 32px;
        color: #52746f;
        font-size: 15px;
        line-height: 1.7;
    }


    /* ========================================================
       PAGE HEADINGS
       ======================================================== */

    .page-kicker {
        color: #087f72;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .page-title {
        color: #123b37;
        font-size: 40px;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .page-description {
        color: #52746f;
        font-size: 16px;
        line-height: 1.7;
        margin-bottom: 25px;
    }


    /* ========================================================
       SELECTBOX
       ======================================================== */

    div[data-baseweb="select"] > div {
        background-color: white !important;
        border: 1px solid #b8cbc7 !important;
        border-radius: 8px !important;
        color: #123b37 !important;
        min-height: 48px !important;
    }

    div[data-baseweb="select"] input {
        color: #123b37 !important;
    }

    div[data-baseweb="select"] span {
        color: #123b37 !important;
    }

    div[data-baseweb="select"] svg {
        fill: #123b37 !important;
    }

    div[data-baseweb="popover"] {
        background-color: white !important;
    }

    div[data-baseweb="menu"] {
        background-color: white !important;
    }

    div[data-baseweb="menu"] div {
        color: #123b37 !important;
    }

    div[data-baseweb="menu"] div:hover {
        background-color: #edf5f3 !important;
    }


    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button {
        background-color: #0b3d39 !important;
        color: white !important;
        border: 1px solid #0b3d39 !important;
        border-radius: 8px !important;
        min-height: 44px !important;
        font-weight: 700 !important;
    }

    .stButton > button:hover {
        background-color: #14534d !important;
        border-color: #14534d !important;
        color: white !important;
    }

    .stButton > button:focus {
        color: white !important;
        border-color: #0b3d39 !important;
        box-shadow: none !important;
        outline: none !important;
    }

    .stButton > button:disabled {
        background-color: #dbe5e2 !important;
        color: #53706b !important;
        border-color: #dbe5e2 !important;
    }


    /* ========================================================
       FILE UPLOADER
       ======================================================== */

    section[data-testid="stFileUploader"] {
        background-color: white !important;
        border-radius: 10px !important;
    }

    section[data-testid="stFileUploader"] > div {
        background-color: white !important;
        border: 1px solid #b8cbc7 !important;
        border-radius: 10px !important;
    }

    section[data-testid="stFileUploader"] div {
        color: #52746f !important;
    }

    section[data-testid="stFileUploader"] small {
        color: #52746f !important;
    }

    section[data-testid="stFileUploader"] button {
        background-color: #0b3d39 !important;
        color: white !important;
        border: 1px solid #0b3d39 !important;
        border-radius: 6px !important;
    }

    section[data-testid="stFileUploader"] button:hover {
        background-color: #14534d !important;
        color: white !important;
    }

    section[data-testid="stFileUploader"] button:focus {
        border-color: #0b3d39 !important;
        box-shadow: none !important;
        outline: none !important;
    }

    section[data-testid="stFileUploader"] input:focus {
        outline: none !important;
        box-shadow: none !important;
    }


    /* ========================================================
       RESULT BOX
       ======================================================== */

    .result-box {
        background-color: white;
        border: 1px solid #d4e1de;
        border-radius: 10px;
        padding: 24px;
        margin-top: 20px;
    }

    .result-label {
        color: #087f72;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .result-value {
        color: #123b37;
        font-size: 28px;
        font-weight: 800;
        line-height: 1.3;
    }

    .result-text {
        color: #52746f;
        font-size: 15px;
        line-height: 1.7;
        margin-top: 10px;
    }


    /* ========================================================
       METRIC CARDS
       ======================================================== */

    .metric-card {
        background-color: white;
        border: 1px solid #d4e1de;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }

    .metric-label {
        color: #52746f;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .metric-value {
        color: #123b37;
        font-size: 26px;
        font-weight: 800;
        margin-top: 6px;
    }


    /* ========================================================
       ALERTS
       ======================================================== */

    .stSuccess {
        background-color: #edf7f3 !important;
        color: #0b5c51 !important;
    }

    .stWarning {
        background-color: #fff8df !important;
        color: #725700 !important;
    }

    .stError {
        background-color: #fff0ed !important;
        color: #8a3225 !important;
    }

    .stInfo {
        background-color: #edf5f3 !important;
        color: #14534d !important;
    }


    /* ========================================================
       MOBILE
       ======================================================== */

    @media (max-width: 900px) {

        .main .block-container {
            padding-left: 1.2rem;
            padding-right: 1.2rem;
            padding-top: 1.2rem;
        }

        .home-title {
            font-size: 36px;
        }

        .page-title {
            font-size: 32px;
        }

    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD SYMPTOM DATA
# ============================================================

@st.cache_data
def load_symptom_data():

    df = pd.read_csv(DATASET_FILE)
    description = pd.read_csv(DESCRIPTION_FILE)
    precaution = pd.read_csv(PRECAUTION_FILE)

    df.columns = df.columns.astype(str).str.strip()
    description.columns = description.columns.astype(str).str.strip()
    precaution.columns = precaution.columns.astype(str).str.strip()

    symptom_columns = [
        column
        for column in df.columns
        if column.lower().startswith("symptom")
    ]

    if not symptom_columns:
        raise ValueError("No symptom columns were found in dataset.csv.")

    df["All_Symptoms"] = df[symptom_columns].fillna("").astype(str).apply(
        lambda row: " ".join(
            value.strip().replace(" ", "_")
            for value in row
            if value.strip()
        ),
        axis=1,
    )

    return df, description, precaution


# ============================================================
# LOAD IMAGE MODELS
# ============================================================

@st.cache_resource
def load_models():

    pneumonia_model = models.resnet18(weights=None)
    pneumonia_model.fc = nn.Linear(
        pneumonia_model.fc.in_features,
        2
    )

    pneumonia_state = torch.load(
        PNEUMONIA_MODEL_FILE,
        map_location=device
    )

    pneumonia_model.load_state_dict(pneumonia_state)
    pneumonia_model = pneumonia_model.to(device)
    pneumonia_model.eval()


    skin_model = models.resnet18(weights=None)
    skin_model.fc = nn.Linear(
        skin_model.fc.in_features,
        7
    )

    skin_state = torch.load(
        SKIN_MODEL_FILE,
        map_location=device
    )

    skin_model.load_state_dict(skin_state)
    skin_model = skin_model.to(device)
    skin_model.eval()

    return pneumonia_model, skin_model


# ============================================================
# TRAIN SYMPTOM MODEL
# ============================================================

@st.cache_resource
def train_symptom_model(df):

    X = df["All_Symptoms"]
    y = df["Disease"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    vectorizer = TfidfVectorizer()

    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    model = LogisticRegression(
        max_iter=500
    )

    model.fit(
        X_train_vectorized,
        y_train
    )

    predictions = model.predict(
        X_test_vectorized
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    return model, vectorizer, accuracy


# ============================================================
# IMAGE TRANSFORM
# ============================================================

image_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        ),
    ]
)


# ============================================================
# SKIN CLASSES
# ============================================================

skin_classes = [
    "bkl",
    "nv",
    "df",
    "mel",
    "vasc",
    "bcc",
    "akiec",
]

skin_class_names = {
    "bkl": "Benign keratosis-like lesions",
    "nv": "Melanocytic nevi",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "vasc": "Vascular lesions",
    "bcc": "Basal cell carcinoma",
    "akiec": "Actinic keratoses / Intraepithelial carcinoma",
}


# ============================================================
# PREDICTION FUNCTIONS
# ============================================================

def predict_xray(model, image):

    image = image.convert("RGB")

    tensor = image_transform(image)
    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(tensor)

        prediction = torch.argmax(
            output,
            dim=1
        ).item()

    classes = [
        "NORMAL",
        "PNEUMONIA"
    ]

    return classes[prediction]


def predict_skin(model, image):

    image = image.convert("RGB")

    tensor = image_transform(image)
    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():

        output = model(tensor)

        prediction = torch.argmax(
            output,
            dim=1
        ).item()

    predicted_class = skin_classes[prediction]

    return skin_class_names[predicted_class]


# ============================================================
# LOAD DATA AND MODELS
# ============================================================

try:

    df, description_df, precaution_df = load_symptom_data()

    symptom_model, vectorizer, symptom_accuracy = train_symptom_model(df)

    pneumonia_model, skin_model = load_models()

except Exception as error:

    st.error(
        "Unable to load the required project files."
    )

    st.exception(error)

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">

            <div class="sidebar-kicker">
                AI-POWERED HEALTH SCREENING
            </div>

            <div class="sidebar-title">
                Health AI Assistant
            </div>

            <div class="sidebar-copy">
                A focused workspace for symptom analysis
                and image-based health screening.
            </div>

        </div>

        <div class="sidebar-divider"></div>
        """,
        unsafe_allow_html=True,
    )

    selected_page = st.radio(
        "Navigation",
        [
            "Overview",
            "Symptoms Checker",
            "Chest X-ray",
            "Skin Cancer",
        ],
        label_visibility="collapsed",
    )

    st.markdown(
        """
        <div class="sidebar-disclaimer">
            This application is an educational AI project.
            Results are intended to support health awareness
            and should not replace professional medical care.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# OVERVIEW
# ============================================================

if selected_page == "Overview":

    left_column, right_column = st.columns(
        [1.1, 0.9],
        gap="large"
    )

    with left_column:

        st.markdown(
            '<div class="home-kicker">PERSONAL HEALTH ASSISTANT</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="home-title">
                AI 3-IN-1 HEALTH CARE SYSTEM
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="home-description">
                Review symptom information, screen a chest X-ray
                for pneumonia, or classify a skin lesion from one
                calm, focused workspace.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="tools-title">AVAILABLE TOOLS</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="feature-item">'
            '<div class="feature-title">'
            'Symptom details'
            '</div>'
            '<div class="feature-copy">'
            'Browse condition descriptions and suggested '
            'precautions from the clinical dataset.'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="feature-item gold">'
            '<div class="feature-title">'
            'Chest X-ray screen'
            '</div>'
            '<div class="feature-copy">'
            'Upload an X-ray image for a pneumonia '
            'classification result.'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="feature-item coral">'
            '<div class="feature-title">'
            'Skin lesion screen'
            '</div>'
            '<div class="feature-copy">'
            'Upload a lesion image for a skin-condition '
            'classification result.'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    with right_column:

        if PROJECT_IMAGE.exists():

            image = Image.open(PROJECT_IMAGE)

            st.image(
                image,
                use_container_width=True
            )

        else:

            st.info(
                "Project image was not found."
            )

        st.markdown(
            """
            <div class="note-box">
                Use the navigation panel to open a tool.
                Results are designed to support, not replace,
                professional medical care.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# SYMPTOMS CHECKER
# ============================================================

elif selected_page == "Symptoms Checker":

    st.markdown(
        '<div class="page-kicker">SYMPTOM ANALYSIS</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-title">Symptoms Checker</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="page-description">
            Select a condition to review its description and
            recommended precautions from the dataset.
        </div>
        """,
        unsafe_allow_html=True,
    )

    diseases = sorted(
        df["Disease"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_disease = st.selectbox(
        "Choose a condition",
        diseases,
        index=None,
        placeholder="Choose a condition to review",
        label_visibility="collapsed",
    )

    if selected_disease:

        description_text = ""
        precautions = []

        description_columns = description_df.columns.tolist()

        if len(description_columns) >= 2:

            disease_column = description_columns[0]
            description_column = description_columns[1]

            matched_description = description_df[
                description_df[disease_column]
                .astype(str)
                .str.strip()
                .str.lower()
                == selected_disease.strip().lower()
            ]

            if not matched_description.empty:

                description_text = str(
                    matched_description.iloc[0][description_column]
                )


        precaution_columns = precaution_df.columns.tolist()

        if len(precaution_columns) >= 2:

            disease_column = precaution_columns[0]

            matched_precaution = precaution_df[
                precaution_df[disease_column]
                .astype(str)
                .str.strip()
                .str.lower()
                == selected_disease.strip().lower()
            ]

            if not matched_precaution.empty:

                row = matched_precaution.iloc[0]

                for column in precaution_columns[1:]:

                    value = row[column]

                    if pd.notna(value) and str(value).strip():

                        precautions.append(
                            str(value).strip()
                        )


        st.markdown(
            """
            <div class="result-box">
                <div class="result-label">
                    Selected condition
                </div>
                <div class="result-value">
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            selected_disease
        )

        st.markdown(
            """
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


        if description_text:

            st.markdown(
                "### Condition Description"
            )

            st.write(
                description_text
            )


        if precautions:

            st.markdown(
                "### Suggested Precautions"
            )

            for precaution in precautions:

                st.markdown(
                    "- " + precaution
                )


# ============================================================
# CHEST X-RAY
# ============================================================

elif selected_page == "Chest X-ray":

    st.markdown(
        '<div class="page-kicker">IMAGE SCREENING</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-title">Chest X-ray Screening</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="page-description">
            Upload a chest X-ray image to screen for a
            pneumonia classification using the trained
            ResNet18 model.
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload chest X-ray",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file:

        image = Image.open(
            uploaded_file
        )

        st.image(
            image,
            caption="Uploaded chest X-ray",
            width=450,
        )

        if st.button(
            "Analyze Chest X-ray",
            use_container_width=True,
        ):

            with st.spinner(
                "Analyzing chest X-ray..."
            ):

                result = predict_xray(
                    pneumonia_model,
                    image
                )

            st.markdown(
                '<div class="result-box">'
                '<div class="result-label">'
                'Screening result'
                '</div>'
                '<div class="result-value">'
                + result +
                '</div>'
                '<div class="result-text">'
                'Model accuracy on the project evaluation set: '
                + XRAY_MODEL_ACCURACY +
                '. This result is for educational screening '
                'and should not be used as a medical diagnosis.'
                '</div>'
                '</div>',
                unsafe_allow_html=True,
            )


# ============================================================
# SKIN CANCER
# ============================================================

elif selected_page == "Skin Cancer":

    st.markdown(
        '<div class="page-kicker">IMAGE SCREENING</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="page-title">Skin Lesion Screening</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="page-description">
            Upload a skin lesion image to classify it into
            one of the trained skin-condition categories.
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload skin lesion image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file:

        image = Image.open(
            uploaded_file
        )

        st.image(
            image,
            caption="Uploaded skin lesion image",
            width=450,
        )

        if st.button(
            "Analyze Skin Lesion",
            use_container_width=True,
        ):

            with st.spinner(
                "Analyzing skin lesion..."
            ):

                result = predict_skin(
                    skin_model,
                    image
                )

            st.markdown(
                '<div class="result-box">'
                '<div class="result-label">'
                'Classification result'
                '</div>'
                '<div class="result-value">'
                + result +
                '</div>'
                '<div class="result-text">'
                'Model accuracy on the project evaluation set: '
                + SKIN_MODEL_ACCURACY +
                '. This result is for educational screening '
                'and should not be used as a medical diagnosis.'
                '</div>'
                '</div>',
                unsafe_allow_html=True,
            )