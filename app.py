import streamlit as st
import pandas as pd
import pickle

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="centered"
)

st.title("💻 Laptop Price Predictor")
st.write("Enter the laptop specifications to estimate its price.")

# ------------------ Load Files ------------------
dataset = pd.read_csv("laptops.csv")

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ------------------ Input Fields ------------------

brand = st.selectbox("Brand", sorted(dataset["brand"].unique()))
processor_brand = st.selectbox(
    "Processor Brand",
    sorted(dataset["processor_brand"].unique())
)

processor_tier = st.selectbox(
    "Processor Tier",
    sorted(dataset["processor_tier"].unique())
)

gpu_brand = st.selectbox(
    "GPU Brand",
    sorted(dataset["gpu_brand"].unique())
)

gpu_type = st.selectbox(
    "GPU Type",
    sorted(dataset["gpu_type"].unique())
)

primary_storage_type = st.selectbox(
    "Primary Storage",
    sorted(dataset["primary_storage_type"].unique())
)

secondary_storage_type = st.selectbox(
    "Secondary Storage",
    sorted(dataset["secondary_storage_type"].unique())
)

os = st.selectbox(
    "Operating System",
    sorted(dataset["OS"].unique())
)

num_cores = st.number_input("CPU Cores", 1, 32, 4)
num_threads = st.number_input("CPU Threads", 1, 64, 8)

ram = st.number_input("RAM (GB)", 2, 128, 8)

primary_storage = st.number_input(
    "Primary Storage Capacity (GB)",
    64,
    4096,
    512
)

secondary_storage = st.number_input(
    "Secondary Storage Capacity (GB)",
    0,
    4096,
    0
)

display = st.number_input(
    "Display Size (inch)",
    10.0,
    20.0,
    15.6
)

resolution_width = st.number_input(
    "Resolution Width",
    1024,
    3840,
    1920
)

resolution_height = st.number_input(
    "Resolution Height",
    768,
    2160,
    1080
)

touch = st.selectbox(
    "Touch Screen",
    [0, 1]
)

rating = st.slider(
    "Rating",
    0.0,
    5.0,
    4.0,
    0.1
)

warranty = st.number_input(
    "Warranty (Years)",
    0,
    5,
    1
)

# ------------------ Prediction ------------------

if st.button("Predict Price"):

    input_df = pd.DataFrame({
        "brand":[brand],
        "Rating":[rating],
        "processor_brand":[processor_brand],
        "processor_tier":[processor_tier],
        "num_cores":[num_cores],
        "num_threads":[num_threads],
        "ram_memory":[ram],
        "primary_storage_type":[primary_storage_type],
        "primary_storage_capacity":[primary_storage],
        "secondary_storage_type":[secondary_storage_type],
        "secondary_storage_capacity":[secondary_storage],
        "gpu_brand":[gpu_brand],
        "gpu_type":[gpu_type],
        "is_touch_screen":[touch],
        "display_size":[display],
        "resolution_width":[resolution_width],
        "resolution_height":[resolution_height],
        "OS":[os],
        "year_of_warranty":[warranty]
    })

    input_df = pd.get_dummies(input_df)

    train_cols = pd.get_dummies(
        dataset.drop(["Price","Model","index"], axis=1)
    ).columns

    input_df = input_df.reindex(columns=train_cols, fill_value=0)

    scaled = scaler.transform(input_df)

    prediction = model.predict(scaled)[0]

    st.success(f"Estimated Laptop Price : ₹ {prediction:,.0f}")
