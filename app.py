import streamlit as st
import numpy as np
import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
import gdown
import zipfile

from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator, Draw

# -------- DOWNLOAD MODELS --------
FILE_ID = "1bv_es0_5a_AvKly15GLGU5qWYewHwRKB"
ZIP_PATH = "models.zip"

if not os.path.exists("models"):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, ZIP_PATH, quiet=False)
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall("models")

# -------- UI --------
st.set_page_config(page_title="Drug Toxicity Predictor", layout="centered")
st.title("Drug Toxicity Prediction (Tox21)")
st.markdown("Predict toxicity from SMILES")

# -------- TARGETS --------
def get_targets():
    files = os.listdir("models")
    return [f.replace("_best.pkl", "") for f in files if f.endswith("_best.pkl")]

TARGETS = get_targets()

# -------- LOAD MODELS --------
@st.cache_resource
def load_models():
    models, scalers = {}, {}

    for t in TARGETS:
        models[t] = joblib.load(f"models/{t}_best.pkl")
        sp = f"models/{t}_scaler.pkl"
        scalers[t] = joblib.load(sp) if os.path.exists(sp) else None

    return models, scalers

models, scalers = load_models()

# -------- FEATURES --------
def get_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None

    desc = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumAromaticRings(mol),
        Descriptors.TPSA(mol),
        mol.GetNumAtoms()
    ]

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp = list(gen.GetFingerprint(mol))

    return np.array(desc + fp), mol

# -------- INPUT --------
smiles = st.text_input("Enter SMILES", "CCO")

# -------- PREDICT --------
if st.button("Predict Toxicity"):

    features, mol = get_features(smiles)

    if features is None:
        st.error("Invalid SMILES")
    else:
        st.success("Valid molecule")

        st.subheader("Molecule Structure")
        st.image(Draw.MolToImage(mol))

        results = []

        st.subheader("Predictions")

        for t in TARGETS:
            model = models[t]
            scaler = scalers[t]

            X = features.reshape(1, -1)
            if scaler:
                X = scaler.transform(X)

            prob = model.predict_proba(X)[0][1]

            results.append({
                "Target": t,
                "Probability": prob,
                "Prediction": "Toxic" if prob > 0.5 else "Non-Toxic"
            })

            if prob > 0.5:
                st.error(f"{t}: Toxic ({prob:.2f})")
            else:
                st.success(f"{t}: Non-Toxic ({prob:.2f})")

        df = pd.DataFrame(results)

        st.subheader("Probability Chart")
        fig, ax = plt.subplots()
        ax.bar(df["Target"], df["Probability"])
        ax.set_ylabel("Probability")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        st.subheader("Download Results")
        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download CSV",
            csv,
            "toxicity_results.csv",
            "text/csv"
        )

st.markdown("---")
st.markdown("Streamlit + RDKit + ML")
