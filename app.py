import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Laptop Price Prediction", layout="centered")

# Load files
model = pickle.load(open("model.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

df = pd.read_csv("laptops.csv")

st.title("💻 Laptop Price Prediction App")

# ---------------- CATEGORICAL COLUMNS ----------------
brand = st.selectbox("Brand", sorted(df["brand"].unique()))
model_name = st.selectbox("Model", sorted(df["Model"].unique()))
processor_brand = st.selectbox("Processor Brand", sorted(df["processor_brand"].unique()))
processor_tier = st.selectbox("Processor Tier", sorted(df["processor_tier"].unique()))
primary_storage_type = st.selectbox("Primary Storage Type", sorted(df["primary_storage_type"].unique()))
secondary_storage_type = st.selectbox("Secondary Storage Type", sorted(df["secondary_storage_type"].unique()))
gpu_brand = st.selectbox("GPU Brand", sorted(df["gpu_brand"].unique()))
gpu_type = st.selectbox("GPU Type", sorted(df["gpu_type"].unique()))
os_name = st.selectbox("OS", sorted(df["OS"].unique()))
year_of_warranty = st.selectbox("Warranty (Years)", sorted(df["year_of_warranty"].unique()))

# ---------------- NUMERICAL COLUMNS ----------------
num_cores = st.number_input("CPU Cores", int(df["num_cores"].min()), int(df["num_cores"].max()), int(df["num_cores"].mean()))
num_threads = st.number_input("CPU Threads", int(df["num_threads"].min()), int(df["num_threads"].max()), int(df["num_threads"].mean()))
ram_memory = st.number_input("RAM (GB)", int(df["ram_memory"].min()), int(df["ram_memory"].max()), int(df["ram_memory"].mean()))
primary_storage_capacity = st.number_input("Primary Storage (GB)", int(df["primary_storage_capacity"].min()), int(df["primary_storage_capacity"].max()), int(df["primary_storage_capacity"].mean()))
secondary_storage_capacity = st.number_input("Secondary Storage (GB)", int(df["secondary_storage_capacity"].min()), int(df["secondary_storage_capacity"].max()), int(df["secondary_storage_capacity"].mean()))
display_size = st.number_input("Display Size", float(df["display_size"].min()), float(df["display_size"].max()), float(df["display_size"].mean()))
resolution_width = st.number_input("Resolution Width", int(df["resolution_width"].min()), int(df["resolution_width"].max()), int(df["resolution_width"].mean()))
resolution_height = st.number_input("Resolution Height", int(df["resolution_height"].min()), int(df["resolution_height"].max()), int(df["resolution_height"].mean()))
is_touch_screen = st.selectbox("Touch Screen", [0, 1])

# ---------------- PREDICTION ----------------
if st.button("Predict Price"):

    input_data = pd.DataFrame([{
        "brand": brand,
        "Model": model_name,
        "processor_brand": processor_brand,
        "processor_tier": processor_tier,
        "primary_storage_type": primary_storage_type,
        "secondary_storage_type": secondary_storage_type,
        "gpu_brand": gpu_brand,
        "gpu_type": gpu_type,
        "OS": os_name,
        "year_of_warranty": year_of_warranty,
        "num_cores": num_cores,
        "num_threads": num_threads,
        "ram_memory": ram_memory,
        "primary_storage_capacity": primary_storage_capacity,
        "secondary_storage_capacity": secondary_storage_capacity,
        "display_size": display_size,
        "resolution_width": resolution_width,
        "resolution_height": resolution_height,
        "is_touch_screen": is_touch_screen
    }])

    # Encode categorical
    cat_cols = [
        "brand","Model","processor_brand","processor_tier",
        "primary_storage_type","secondary_storage_type",
        "gpu_brand","gpu_type","OS","year_of_warranty"
    ]

    encoded = encoder.transform(input_data[cat_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))

    # Numeric part
    num_df = input_data.drop(columns=cat_cols)

    # Final merge
    final_df = pd.concat([encoded_df, num_df.reset_index(drop=True)], axis=1)

    # Scale
    final_scaled = scaler.transform(final_df)

    # Predict
    prediction = model.predict(final_scaled)

    st.success(f"💰 Predicted Laptop Price: ₹{prediction[0]:,.2f}")
