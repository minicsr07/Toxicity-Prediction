import streamlit as st
import numpy as np
import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator, Draw

# PAGE CONFIG
st.set_page_config(
    page_title="Drug Toxicity Predictor",
    layout="centered"
)

st.title("Drug Toxicity Prediction (Tox21)")
st.markdown("Predict toxicity from molecular structure (SMILES)")


# LOAD TARGETS

def get_targets():
    files = os.listdir("models")
    return [f.replace("_best.pkl", "") for f in files if f.endswith("_best.pkl")]

TARGETS = get_targets()

# LOAD MODELS

@st.cache_resource
def load_models():
    models = {}
    scalers = {}

    for target in TARGETS:
        models[target] = joblib.load(f"models/{target}_best.pkl")

        scaler_path = f"models/{target}_scaler.pkl"
        if os.path.exists(scaler_path):
            scalers[target] = joblib.load(scaler_path)
        else:
            scalers[target] = None

    return models, scalers

models, scalers = load_models()


# FEATURE EXTRACTION

def get_features(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None, None

    descriptors = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumAromaticRings(mol),
        Descriptors.TPSA(mol),
        mol.GetNumAtoms()
    ]

    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp = list(morgan_gen.GetFingerprint(mol))

    return np.array(descriptors + fp), mol


# INPUT

smiles_input = st.text_input("Enter SMILES", "CCO")


# PREDICT BUTTON

if st.button("Predict Toxicity"):

    features, mol = get_features(smiles_input)

    if features is None:
        st.error("Invalid SMILES string")
    else:
        st.success("Valid molecule")

        # -----------------------------
        # SHOW MOLECULE IMAGE
        # -----------------------------
        st.subheader("Molecule Structure")
        img = Draw.MolToImage(mol)
        st.image(img)

        results = []

        st.subheader("Toxicity Predictions")

        for target in TARGETS:
            model = models[target]
            scaler = scalers[target]

            X = features.reshape(1, -1)

            if scaler is not None:
                X = scaler.transform(X)

            prob = model.predict_proba(X)[0][1]

            results.append({
                "Target": target,
                "Probability": prob,
                "Prediction": "Toxic" if prob > 0.5 else "Non-Toxic"
            })

            if prob > 0.5:
                st.error(f"{target}: Toxic ({prob:.2f})")
            else:
                st.success(f"{target}: Non-Toxic ({prob:.2f})")

        # -----------------------------
        # CONVERT TO DATAFRAME
        # -----------------------------
        df_results = pd.DataFrame(results)

        # -----------------------------
        # BAR CHART
        # -----------------------------
        st.subheader("Toxicity Probability Chart")

        fig, ax = plt.subplots()
        ax.bar(df_results["Target"], df_results["Probability"])
        ax.set_ylabel("Probability")
        ax.set_title("Toxicity Probability per Target")
        plt.xticks(rotation=45)

        st.pyplot(fig)

        # -----------------------------
        # DOWNLOAD CSV
        # -----------------------------
        st.subheader("Download Results")

        csv = df_results.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="Download Predictions CSV",
            data=csv,
            file_name="toxicity_results.csv",
            mime="text/csv"
        )

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown("Built with Streamlit + RDKit + ML Models")
