import streamlit as st
import pandas as pd
import pickle

# Load files
model = pickle.load(open("laptop_price_model1.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Dataset
df = pd.read_csv("laptops.csv")

st.set_page_config(page_title="Laptop Price Prediction")

st.title("💻 Laptop Price Prediction")

# Inputs
brand = st.selectbox("Brand", sorted(df["Brand"].unique()))
processor = st.selectbox("Processor", sorted(df["Processor"].unique()))
ram = st.selectbox("RAM", sorted(df["RAM"].unique()))
storage = st.selectbox("Storage", sorted(df["Storage"].unique()))
gpu = st.selectbox("GPU", sorted(df["GPU"].unique()))
os = st.selectbox("Operating System", sorted(df["OS"].unique()))

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
        "Brand":[brand],
        "Processor":[processor],
        "RAM":[ram],
        "Storage":[storage],
        "GPU":[gpu],
        "OS":[os],
        "Display Size":[display],
        "Rating":[rating]
    })

    # Encode categorical columns
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

    # Scale
    final_df = scaler.transform(final_df)

    # Prediction
    price = model.predict(final_df)

    st.success(f"💰 Estimated Laptop Price: ₹{price[0]:,.2f}")
