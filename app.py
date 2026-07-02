import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="Laptop Price Prediction")

# Check files
required_files = [
    "model.pkl",
    "encoder.pkl",
    "scaler.pkl",
    "laptops.csv"
]

for file in required_files:
    if not os.path.exists(file):
        st.error(f"Missing file: {file}")
        st.stop()

# Load model
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("encoder.pkl", "rb") as f:
        encoder = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

except Exception as e:
    st.error(f"Error loading model files:\n\n{e}")
    st.stop()

# Load dataset
df = pd.read_csv("laptops.csv")

st.title("💻 Laptop Price Prediction")

brand = st.selectbox("Brand", sorted(df["Brand"].unique()))
processor = st.selectbox("Processor", sorted(df["Processor"].unique()))
ram = st.selectbox("RAM", sorted(df["RAM"].unique()))
storage = st.selectbox("Storage", sorted(df["Storage"].unique()))
gpu = st.selectbox("GPU", sorted(df["GPU"].unique()))
os_name = st.selectbox("Operating System", sorted(df["OS"].unique()))

display = st.number_input(
    "Display Size (inches)",
    float(df["Display Size"].min()),
    float(df["Display Size"].max()),
    float(df["Display Size"].mean())
)

rating = st.number_input(
    "Rating",
    float(df["Rating"].min()),
    float(df["Rating"].max()),
    float(df["Rating"].mean())
)

if st.button("Predict Price"):

    input_df = pd.DataFrame({
        "Brand": [brand],
        "Processor": [processor],
        "RAM": [ram],
        "Storage": [storage],
        "GPU": [gpu],
        "OS": [os_name],
        "Display Size": [display],
        "Rating": [rating]
    })

    cat_cols = input_df.select_dtypes(include="object").columns

    encoded = encoder.transform(input_df[cat_cols])

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(cat_cols)
    )

    input_df = input_df.drop(columns=cat_cols)

    final_df = pd.concat(
        [input_df.reset_index(drop=True), encoded_df],
        axis=1
    )

    final_df = scaler.transform(final_df)

    price = model.predict(final_df)

    st.success(f"💰 Estimated Laptop Price: ₹{price[0]:,.2f}")
