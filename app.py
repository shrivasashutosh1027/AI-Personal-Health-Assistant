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


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="CareSight | Health AI",
    page_icon="+",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# GLOBAL SETTINGS
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

XRAY_MODEL_ACCURACY = "84.29%"
SKIN_MODEL_ACCURACY = "74.47%"


# =========================================================
# LOAD SYMPTOM DATA
# =========================================================

@st.cache_data(show_spinner=False)
def load_symptom_data():

    df = pd.read_csv("dataset.csv")
    desc = pd.read_csv("symptom_Description.csv")
    prec = pd.read_csv("symptom_precaution.csv")

    df.columns = df.columns.str.strip()
    desc.columns = desc.columns.str.strip()
    prec.columns = prec.columns.str.strip()

    symptom_cols = [
        column
        for column in df.columns
        if "Symptom" in column
    ]

    df["All_Symptoms"] = df[symptom_cols].apply(
        lambda row: " ".join(
            str(symptom)
            .strip()
            .replace(" ", "_")
            for symptom in row
            if pd.notna(symptom)
            and str(symptom).strip()
        ),
        axis=1,
    )

    return df, desc, prec


# =========================================================
# LOAD DEEP LEARNING MODELS
# =========================================================

@st.cache_resource(
    show_spinner="Loading diagnostic models..."
)
def load_models():

    # -----------------------------------------------------
    # Chest X-ray model
    # -----------------------------------------------------

    xray_model = models.resnet18(
        pretrained=True
    )

    xray_model.fc = nn.Linear(
        xray_model.fc.in_features,
        2
    )

    xray_model.load_state_dict(
        torch.load(
            "pneumonia_model.pth",
            map_location=device
        )
    )

    xray_model = xray_model.to(device)
    xray_model.eval()

    # -----------------------------------------------------
    # Skin cancer model
    # -----------------------------------------------------

    skin_model = models.resnet18(
        pretrained=True
    )

    skin_model.fc = nn.Linear(
        skin_model.fc.in_features,
        7
    )

    skin_model.load_state_dict(
        torch.load(
            "skin_cancer_model.pth",
            map_location=device
        )
    )

    skin_model = skin_model.to(device)
    skin_model.eval()

    return xray_model, skin_model


# =========================================================
# LOAD DATA
# =========================================================

df, desc, prec = load_symptom_data()


# =========================================================
# TRAIN SYMPTOM MODEL
# =========================================================

X = df["All_Symptoms"]
y = df["Disease"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(
    X_train
)

lr_model = LogisticRegression(
    max_iter=500
)

lr_model.fit(
    X_train_tfidf,
    y_train
)

symptom_predictions = lr_model.predict(
    vectorizer.transform(X_test)
)

SYMPTOM_MODEL_ACCURACY = (
    f"{accuracy_score(y_test, symptom_predictions) * 100:.2f}%"
)


# =========================================================
# IMAGE TRANSFORMS
# =========================================================

xray_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5]
    ),
])

skin_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5]
    ),
])


# =========================================================
# SKIN CANCER CLASSES
# =========================================================

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


# =========================================================
# CHEST X-RAY PREDICTION
# =========================================================

def predict_xray(img):

    xray_model, _ = load_models()

    img_tensor = (
        xray_transform(
            img.convert("RGB")
        )
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():

        output = xray_model(
            img_tensor
        )

        predicted = torch.argmax(
            output,
            dim=1
        )

    return [
        "NORMAL",
        "PNEUMONIA"
    ][predicted.item()]


# =========================================================
# SKIN LESION PREDICTION
# =========================================================

def predict_skin(img):

    _, skin_model = load_models()

    img_tensor = (
        skin_transform(
            img.convert("RGB")
        )
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():

        output = skin_model(
            img_tensor
        )

        predicted = torch.argmax(
            output,
            dim=1
        )

    return skin_class_names[
        skin_classes[predicted.item()]
    ]


# =========================================================
# CARESIGHT CSS
# =========================================================

st.markdown(
    """
<style>

/* =========================================================
   CARESIGHT COLOR SYSTEM
   ========================================================= */

:root {
    --ink: #173230;
    --muted: #5e7773;
    --paper: #f5f8f6;
    --line: #d6e3de;

    --teal: #0b7367;
    --teal-dark: #07584f;

    --green-dark: #0b3d39;
    --green-light: #e8f3f0;

    --gold: #e9b44c;
    --coral: #d96850;

    --white: #ffffff;
}


/* =========================================================
   MAIN APPLICATION
   ========================================================= */

.stApp {
    background: var(--paper) !important;
    color: var(--ink) !important;
    font-family: "DM Sans", sans-serif;
}

#MainMenu {
    visibility: hidden !important;
}

footer {
    visibility: hidden !important;
}

header {
    background: transparent !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
}

[data-testid="stToolbar"] {
    display: none !important;
}


/* =========================================================
   MAIN CONTAINER
   ========================================================= */

[data-testid="stAppViewContainer"] > .main {
    padding-top: 1.25rem;
}

[data-testid="stMainBlockContainer"] {
    max-width: 1240px;
    padding: 0 2.1rem 3rem;
}


/* =========================================================
   SIDEBAR
   ========================================================= */

[data-testid="stSidebar"] {
    background: var(--green-dark) !important;
    border-right: 1px solid #22534e !important;
}

[data-testid="stSidebar"] > div:first-child {
    background: var(--green-dark) !important;
}

[data-testid="stSidebarContent"] {
    background: var(--green-dark) !important;
}

[data-testid="stSidebarHeader"] {
    background: var(--green-dark) !important;
}


/* =========================================================
   SIDEBAR TEXT
   ========================================================= */

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label {
    color: #f4fbf8 !important;
}


/* =========================================================
   SIDEBAR BRAND
   ========================================================= */

.brand-kicker {
    color: #9bd3c5 !important;
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.2rem;
}

.brand-name {
    color: #ffffff !important;
    font-family: "Playfair Display", serif;
    font-size: 2rem;
    line-height: 1;
    margin: 0 0 0.45rem;
}

.brand-copy {
    color: #c2dad4 !important;
    font-size: 0.86rem;
    line-height: 1.55;
    margin-bottom: 1.75rem;
}


/* =========================================================
   SIDEBAR NAVIGATION
   ========================================================= */

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    color: #f4fbf8 !important;
    font-weight: 500 !important;
    padding: 0.2rem 0 !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label p {
    color: #f4fbf8 !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label span {
    color: #f4fbf8 !important;
}


/* =========================================================
   SIDEBAR DISCLAIMER
   ========================================================= */

[data-testid="stSidebar"]
[data-testid="stCaptionContainer"] {
    color: #c2dad4 !important;
}

[data-testid="stSidebar"]
[data-testid="stCaptionContainer"] p {
    color: #c2dad4 !important;
}


/* =========================================================
   SIDEBAR BEHAVIOR
   ========================================================= */

/* Laptop: permanently open. Mobile: native Streamlit toggle remains available. */

@media (min-width: 701px) {
    [data-testid="stSidebar"] {
        transform: translateX(0) !important;
        visibility: visible !important;
        width: 340px !important;
        min-width: 340px !important;
        max-width: 340px !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        width: 340px !important;
    }

    [data-testid="stSidebarCollapseButton"],
    [data-testid="stExpandSidebarButton"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] {
        display: none !important;
    }
}

@media (max-width: 700px) {
    /* The native Streamlit header must remain available because the
       collapsed-sidebar open control is rendered there. */
    header[data-testid="stHeader"],
    [data-testid="stHeader"] {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        z-index: 999999 !important;
        pointer-events: auto !important;
        background: transparent !important;
    }

    /* Closed sidebar: keep every native expand-control variant visible. */
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"],
    [data-testid="stExpandSidebarButton"] {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        pointer-events: auto !important;
        z-index: 2147483647 !important;
    }

    [data-testid="stSidebarCollapsedControl"] button,
    [data-testid="collapsedControl"] button,
    [data-testid="stExpandSidebarButton"] button,
    [data-testid="stSidebarCollapseButton"] button {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        align-items: center !important;
        justify-content: center !important;
        width: 40px !important;
        height: 40px !important;
        min-width: 40px !important;
        min-height: 40px !important;
        padding: 0 !important;
        margin: 0 !important;
        background: var(--green-dark) !important;
        color: #ffffff !important;
        border: 1px solid #5c8f87 !important;
        border-radius: 7px !important;
        box-shadow: none !important;
        outline: none !important;
    }

    /* Open sidebar: keep the native close button visible. */
    [data-testid="stSidebarCollapseButton"] {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        z-index: 2147483647 !important;
    }

    [data-testid="stSidebarCollapseButton"] button svg,
    [data-testid="stSidebarCollapsedControl"] button svg,
    [data-testid="collapsedControl"] button svg,
    [data-testid="stExpandSidebarButton"] button svg {
        width: 20px !important;
        height: 20px !important;
        color: #ffffff !important;
        fill: #ffffff !important;
        stroke: #ffffff !important;
        opacity: 1 !important;
    }

    [data-testid="stSidebarCollapseButton"] button:hover,
    [data-testid="stSidebarCollapsedControl"] button:hover,
    [data-testid="collapsedControl"] button:hover,
    [data-testid="stExpandSidebarButton"] button:hover {
        background: var(--teal-dark) !important;
        border-color: #82b6ad !important;
    }

    [data-testid="stSidebarCollapseButton"] button:focus,
    [data-testid="stSidebarCollapsedControl"] button:focus,
    [data-testid="collapsedControl"] button:focus,
    [data-testid="stExpandSidebarButton"] button:focus {
        outline: none !important;
        box-shadow: none !important;
    }
}

@media (max-width: 400px) {
    [data-testid="stSidebarCollapseButton"] button,
    [data-testid="stSidebarCollapsedControl"] button,
    [data-testid="collapsedControl"] button,
    [data-testid="stExpandSidebarButton"] button {
        width: 38px !important;
        height: 38px !important;
        min-width: 38px !important;
        min-height: 38px !important;
    }
}

/* =========================================================
   PAGE TYPOGRAPHY
   ========================================================= */

.page-kicker {
    color: var(--teal) !important;

    font-size: 0.78rem;

    font-weight: 700;

    letter-spacing: 0.12em;

    text-transform: uppercase;

    margin-bottom: 0.6rem;
}

.page-title {
    color: var(--ink) !important;

    font-family: "Playfair Display", serif;

    font-size: clamp(
        2.1rem,
        4vw,
        3.65rem
    );

    line-height: 1.08;

    margin: 0 0 0.8rem;
}

.page-copy {
    color: var(--ink) !important;

    font-size: 1.04rem;

    line-height: 1.65;

    max-width: 42rem;

    margin: 0 0 1.7rem;
}

.hero-rule {
    border: 0;

    border-top: 1px solid var(--line);

    margin: 1.35rem 0 1.75rem;
}

.section-label {
    color: var(--teal) !important;

    font-size: 0.78rem;

    font-weight: 700;

    letter-spacing: 0.1em;

    text-transform: uppercase;

    margin-bottom: 0.35rem;
}


/* =========================================================
   OVERVIEW FEATURE ITEMS
   ========================================================= */

.feature-item {
    border-top: 3px solid var(--teal);

    padding: 1rem 0 0.8rem;

    margin-bottom: 1rem;
}

.feature-item.gold {
    border-color: var(--gold);
}

.feature-item.coral {
    border-color: var(--coral);
}

.feature-title {
    color: var(--ink) !important;

    font-size: 1.05rem;

    font-weight: 700;

    margin-bottom: 0.25rem;
}

.feature-copy {
    color: var(--muted) !important;

    font-size: 0.93rem;

    line-height: 1.5;
}


/* =========================================================
   QUIET NOTE
   ========================================================= */

.quiet-note {
    border-left: 3px solid var(--gold);

    color: var(--muted) !important;

    font-size: 0.9rem;

    line-height: 1.55;

    padding:
        0.65rem
        0
        0.65rem
        1rem;

    margin-top: 1rem;
}


/* =========================================================
   SELECTBOX
   ========================================================= */

/* Main selectbox container */

[data-testid="stSelectbox"] {
    color: var(--ink) !important;
}


/* Label */

[data-testid="stSelectbox"] label {
    color: var(--ink) !important;
    opacity: 1 !important;
}


/* Selectbox outer element */

[data-testid="stSelectbox"]
[data-baseweb="select"] {
    background-color: #ffffff !important;

    border: 1px solid #b8cec7 !important;

    border-radius: 7px !important;

    color: #173230 !important;

    opacity: 1 !important;

    box-shadow: none !important;
}


/* Selectbox inner elements */

[data-testid="stSelectbox"]
[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    color: #173230 !important;
}


/* Every text node */

[data-testid="stSelectbox"]
[data-baseweb="select"] div {
    color: #173230 !important;
}


/* Span */

[data-testid="stSelectbox"]
[data-baseweb="select"] span {
    color: #173230 !important;

    opacity: 1 !important;
}


/* Button-like area */

[data-testid="stSelectbox"]
[role="button"] {
    background-color: #ffffff !important;

    color: #173230 !important;

    opacity: 1 !important;
}


/* Text in button */

[data-testid="stSelectbox"]
[role="button"] span {
    color: #173230 !important;

    opacity: 1 !important;
}


/* Input */

[data-testid="stSelectbox"] input {
    color: #173230 !important;

    -webkit-text-fill-color: #173230 !important;

    opacity: 1 !important;
}


/* Input placeholder */

[data-testid="stSelectbox"]
input::placeholder {
    color: #5e7773 !important;

    -webkit-text-fill-color: #5e7773 !important;

    opacity: 1 !important;
}


/* BaseWeb placeholder */

[data-testid="stSelectbox"]
[data-baseweb="select"]
[aria-selected="false"] {
    color: #5e7773 !important;
}


/* SVG arrow */

[data-testid="stSelectbox"]
svg {
    color: #0b7367 !important;

    fill: #0b7367 !important;

    stroke: #0b7367 !important;
}


/* Selectbox focus */

[data-testid="stSelectbox"]
[data-baseweb="select"]:focus-within {
    border-color: #0b7367 !important;

    box-shadow:
        0 0 0 1px
        #0b7367 !important;

    outline: none !important;
}


/* =========================================================
   SELECTBOX DROPDOWN
   ========================================================= */

[data-baseweb="popover"] {
    background: #ffffff !important;
}

[data-baseweb="popover"] div {
    color: #173230 !important;
}

[data-baseweb="popover"] span {
    color: #173230 !important;
}

[role="listbox"] {
    background: #ffffff !important;

    color: #173230 !important;
}

[role="option"] {
    background: #ffffff !important;

    color: #173230 !important;
}

[role="option"] span {
    color: #173230 !important;
}

[role="option"]:hover {
    background: #e8f3f0 !important;

    color: #07584f !important;
}


/* =========================================================
   BUTTONS
   ========================================================= */

.stButton > button {
    width: 100%;

    min-height: 2.85rem;

    background: var(--teal) !important;

    color: #ffffff !important;

    border: none !important;

    border-radius: 6px !important;

    font-weight: 700;
}

.stButton > button p {
    color: #ffffff !important;
}

.stButton > button span {
    color: #ffffff !important;
}

.stButton > button:hover {
    background: var(--teal-dark) !important;

    color: #ffffff !important;
}


/* Disabled button */

.stButton > button:disabled {
    background: #d5e1dd !important;

    color: #173230 !important;

    opacity: 1 !important;

    cursor: not-allowed !important;
}

.stButton > button:disabled p {
    color: #173230 !important;
}

.stButton > button:disabled span {
    color: #173230 !important;
}


/* =========================================================
   FILE UPLOADER
   ========================================================= */

[data-testid="stFileUploader"] {
    width: 100% !important;
}


/* Uploader box */

[data-testid="stFileUploaderDropzone"] {
    background: #ffffff !important;

    border: 1px solid #b8cec7 !important;

    border-radius: 7px !important;

    padding: 1.6rem 1rem !important;

    box-shadow: none !important;

    outline: none !important;
}


/* Prevent red focus border */

[data-testid="stFileUploaderDropzone"]:focus,
[data-testid="stFileUploaderDropzone"]:focus-within,
[data-testid="stFileUploaderDropzone"]:active {
    background: #ffffff !important;

    border: 1px solid #b8cec7 !important;

    outline: none !important;

    box-shadow: none !important;
}


/* Hover */

[data-testid="stFileUploaderDropzone"]:hover {
    background: #fbfdfc !important;

    border-color: #0b7367 !important;
}


/* Uploader instruction area */

[data-testid="stFileUploaderDropzoneInstructions"] {
    color: #5e7773 !important;

    opacity: 1 !important;
}


/* Drag and drop text */

[data-testid="stFileUploaderDropzoneInstructions"] span {
    color: #5e7773 !important;

    opacity: 1 !important;
}


/* 200MB text */

[data-testid="stFileUploaderDropzoneInstructions"] small {
    color: #5e7773 !important;

    opacity: 1 !important;
}


/* Generic uploader text */

[data-testid="stFileUploaderDropzone"] p {
    color: #5e7773 !important;
}

[data-testid="stFileUploaderDropzone"] span {
    color: #5e7773 !important;
}


/* Browse files button */

[data-testid="stFileUploaderDropzone"] button {
    background: #0b3d39 !important;

    color: #ffffff !important;

    border: none !important;

    border-radius: 6px !important;

    font-weight: 600 !important;
}

[data-testid="stFileUploaderDropzone"] button:hover {
    background: #07584f !important;
}

[data-testid="stFileUploaderDropzone"] button span {
    color: #ffffff !important;
}


/* Upload icon */

[data-testid="stFileUploaderDropzone"] svg {
    color: #5e7773 !important;

    fill: #5e7773 !important;

    stroke: #5e7773 !important;
}


/* =========================================================
   ALERT BOXES
   ========================================================= */

[data-testid="stAlert"] {
    border-radius: 6px !important;

    color: var(--ink) !important;
}

[data-testid="stAlert"] p {
    color: var(--ink) !important;
}

[data-testid="stAlert"] span {
    color: var(--ink) !important;
}


/* =========================================================
   IMAGE
   ========================================================= */

[data-testid="stImage"] img {
    border-radius: 6px;

    border: 1px solid var(--line);
}

[data-testid="stImage"] figcaption {
    color: var(--muted) !important;
}


/* =========================================================
   GENERAL TEXT
   ========================================================= */

[data-testid="stAppViewContainer"]
.main label {
    color: var(--ink) !important;
}

[data-testid="stMarkdownContainer"] p {
    color: var(--ink);
}

[data-testid="stMarkdownContainer"] li {
    color: var(--ink);
}


/* =========================================================
   RESPONSIVE
   ========================================================= */

@media (max-width: 1100px) {

    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;

        gap: 1rem !important;
    }

    [data-testid="stHorizontalBlock"]
    > [data-testid="stColumn"] {
        flex: 1 1 100% !important;

        min-width: 100% !important;
    }
}


@media (max-width: 700px) {

    [data-testid="stMainBlockContainer"] {
        max-width: 100%;

        padding:
            0
            1rem
            2.5rem;

        overflow-x: hidden;
    }

    [data-testid="stAppViewContainer"] > .main {
        padding-top: 0.8rem;
    }

    .page-title {
        font-size: 2.35rem;
    }

    [data-testid="stSidebar"] {
        width: min(86vw, 18rem) !important;

        min-width: min(86vw, 18rem) !important;
    }

}

</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        '<div class="brand-kicker">'
        'AI-POWERED HEALTH SCREENING'
        '</div>'
        '<div class="brand-name">'
        'Health AI Assistant'
        '</div>'
        '<div class="brand-copy">'
        'A focused workspace for symptom details '
        'and image-based screening.'
        '</div>',
        unsafe_allow_html=True,
    )

    module = st.radio(
        "Choose a workspace",
        [
            "Overview",
            "Symptoms Checker",
            "Chest X-ray",
            "Skin Cancer",
        ],
        label_visibility="collapsed",
    )

    st.caption(
        "For informational screening only. "
        "Consult a qualified clinician for diagnosis "
        "or treatment."
    )


# =========================================================
# OVERVIEW
# =========================================================

if module == "Overview":

    intro_col, image_col = st.columns(
        [1.05, 0.95],
        gap="large"
    )

    with intro_col:

        st.markdown(
            """
            <div class="page-kicker">
                Personal health assistant
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <h1 class="page-title">
                AI 3-IN-1 HEALTH 
                CARE SYSTEM
            </h1>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <p class="page-copy">
                Review symptom information, screen a chest
                X-ray for pneumonia, or classify a skin lesion
                from one calm, focused workspace.
            </p>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <hr class="hero-rule">
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="section-label">
                Available tools
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Symptom feature
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

        # X-ray feature
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

        # Skin feature
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

    with image_col:

        st.image(
            "ChatGPT Image Aug 28, 2025, 02_40_38 AM.png",
            use_container_width=True,
        )

        st.markdown(
            """
            <div class="quiet-note">
                Use the navigation panel to open a tool.
                Results are designed to support, not replace,
                professional medical care.
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# SYMPTOMS CHECKER
# =========================================================

elif module == "Symptoms Checker":

    st.markdown(
        """
        <div class="page-kicker">
            Clinical reference
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <h1 class="page-title">
            Symptom and condition details
        </h1>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <p class="page-copy">
            Select a condition to review its dataset
            description and practical precautions.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-label">
            Condition
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Condition selector
    selected_disease = st.selectbox(
        "Choose a condition",
        sorted(
            df["Disease"]
            .unique()
            .tolist()
        ),
        index=None,
        placeholder="Choose a condition to review",
        label_visibility="collapsed",
    )

    # Details button
    if st.button(
        "View condition details",
        disabled=selected_disease is None,
    ):

        description = desc[
            desc["Disease"] == selected_disease
        ]["Description"].values

        if len(description) > 0:
            description = description[0]
        else:
            description = "No description available."

        precaution_data = prec[
            prec["Disease"] == selected_disease
        ]

        if not precaution_data.empty:

            precautions = (
                precaution_data
                .iloc[0, 1:]
                .dropna()
                .tolist()
            )

        else:

            precautions = [
                "No precaution listed."
            ]

        st.success(
            f"Condition selected: {selected_disease}"
        )

        details_col, precaution_col = st.columns(
            [1.15, 0.85],
            gap="large"
        )

        with details_col:

            st.markdown(
                """
                <div class="section-label">
                    Overview
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write(description)

        with precaution_col:

            st.markdown(
                """
                <div class="section-label">
                    Suggested precautions
                </div>
                """,
                unsafe_allow_html=True,
            )

            for item in precautions:

                st.write(
                    f"- {item}"
                )


# =========================================================
# CHEST X-RAY
# =========================================================

elif module == "Chest X-ray":

    st.markdown(
        """
        <div class="page-kicker">
            Image screening
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <h1 class="page-title">
            Chest X-ray review
        </h1>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <p class="page-copy">
            Upload a clear chest X-ray image to screen
            for a normal or pneumonia classification.
        </p>
        """,
        unsafe_allow_html=True,
    )

    upload_col, preview_col = st.columns(
        [0.9, 1.1],
        gap="large"
    )

    with upload_col:

        uploaded_file = st.file_uploader(
            "Upload chest X-ray",
            type=[
                "jpg",
                "png",
                "jpeg",
            ],
            help="Accepted formats: JPG, PNG, JPEG.",
        )

        st.markdown(
            """
            <div class="quiet-note">
                Choose a well-lit, uncropped image.
                The output is a screening classification,
                not a diagnosis.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with preview_col:

        if uploaded_file:

            img = Image.open(
                uploaded_file
            )

            st.image(
                img,
                caption="Uploaded chest X-ray",
                use_container_width=True,
            )

            if st.button(
                "Run pneumonia screening"
            ):

                with st.spinner(
                    "Reviewing image..."
                ):

                    result = predict_xray(
                        img
                    )

                st.success(
                    f"Screening result: {result}"
                )

        else:

            st.info(
                "Your uploaded image preview and "
                "screening result will appear here."
            )


# =========================================================
# SKIN CANCER
# =========================================================

elif module == "Skin Cancer":

    st.markdown(
        """
        <div class="page-kicker">
            Image screening
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <h1 class="page-title">
            Skin lesion review
        </h1>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <p class="page-copy">
            Upload a clear image of a skin lesion to
            receive a classification from the skin model.
        </p>
        """,
        unsafe_allow_html=True,
    )

    upload_col, preview_col = st.columns(
        [0.9, 1.1],
        gap="large"
    )

    with upload_col:

        uploaded_file = st.file_uploader(
            "Upload skin lesion image",
            type=[
                "jpg",
                "png",
                "jpeg",
            ],
            help="Accepted formats: JPG, PNG, JPEG.",
        )

        st.markdown(
            """
            <div class="quiet-note">
                A clear, close, evenly lit image produces
                the most useful screening output. Seek
                clinical advice for any changing or
                concerning lesion.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with preview_col:

        if uploaded_file:

            img = Image.open(
                uploaded_file
            )

            st.image(
                img,
                caption="Uploaded skin lesion",
                use_container_width=True,
            )

            if st.button(
                "Run skin lesion screening"
            ):

                with st.spinner(
                    "Reviewing image..."
                ):

                    result = predict_skin(
                        img
                    )

                st.success(
                    f"Screening result: {result}"
                )

        else:

            st.info(
                "Your uploaded image preview and "
                "screening result will appear here."
            )