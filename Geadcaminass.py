import streamlit as st
import torch
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
import pandas as pd

# GradCAM Implementation
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks to capture activations and gradients
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_tensor, target_class):
        # Zero gradients
        self.model.zero_grad()

        # Forward pass
        output = self.model(input_tensor)

        # Backward pass
        class_score = output[0, target_class]
        class_score.backward()

        # Compute GradCAM heatmap
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze()
        cam = torch.clamp(cam, min=0).cpu().numpy()  # ReLU
        cam = cam / np.max(cam)  # Normalize to [0, 1]
        return cam

    def get_activation_coordinates(self, heatmap, threshold):
        """Extract coordinates of activation points from the heatmap based on a threshold."""
        coordinates = []
        height, width = heatmap.shape
        for y in range(height):
            for x in range(width):
                if heatmap[y, x] >= threshold:
                    coordinates.append((x, y))
        return coordinates

# Define preprocessing using EfficientNet weights
def preprocess_image(image):
    preprocess = EfficientNet_V2_S_Weights.IMAGENET1K_V1.transforms()
    return preprocess(image).unsqueeze(0)  # Add batch dimension

# Overlay GradCAM heatmap on the image
def overlay_heatmap(image, heatmap):
    # Resize heatmap to match image dimensions
    heatmap_resized = cv2.resize(heatmap, (image.width, image.height))
    heatmap_resized = np.uint8(255 * heatmap_resized)

    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Blend heatmap with the original image
    overlay = Image.blend(image, Image.fromarray(heatmap_colored), alpha=0.5)
    return overlay

# Streamlit UI
st.set_page_config(page_title="MRI Tumor Classifier", page_icon="🌐", layout="wide")
st.title("MRI Brain Tumor Classification Ensemble Model with GradCAM")
st.write("Fill in the patient information and upload an MRI image to classify it and visualize the GradCAM heatmap.")

# Patient information form
with st.sidebar:
    st.header("Patient Information")
    with st.form("patient_form"):
        patient_name = st.text_input("Patient Name", "")
        patient_age = st.number_input("Patient Age", min_value=0, max_value=120, step=1)
        patient_id = st.text_input("Patient ID", "")
        mri_test_date = st.date_input("MRI Test Date")
        gender = st.radio("Gender", ("Male", "Female"))
        provinces = ["Aceh", "Bali", "Banten", "Bengkulu", "Central Java", "Central Kalimantan", "Central Sulawesi", "East Java", "East Kalimantan", "East Nusa Tenggara", "Gorontalo", "Jakarta", "Jambi", "Lampung", "Maluku", "North Kalimantan", "North Maluku", "North Sulawesi", "North Sumatra", "Papua", "Riau", "Riau Islands", "Southeast Sulawesi", "South Kalimantan", "South Sulawesi", "South Sumatra", "West Java", "West Kalimantan", "West Nusa Tenggara", "West Papua", "West Sulawesi", "West Sumatra", "Yogyakarta"]
        selected_province = st.selectbox("Province", provinces)
        submitted = st.form_submit_button("Save Information")

    if submitted:
        st.success("Patient information saved. Please upload an image for prediction.")

# Load model
model_path = "model.pt"
model = efficientnet_v2_s(weights=None)  # Define base architecture
num_classes = 4  # Adjust for your task
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)  # Modify final layer
model.load_state_dict(torch.load(model_path))  # Load trained weights
model.eval()  # Set model to evaluation mode

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(Image.open(uploaded_file), caption="Uploaded Image", use_column_width=True)
    if st.button("Predict"):
        image = Image.open(uploaded_file).convert("RGB")

        # Preprocess the image
        input_tensor = preprocess_image(image)

        # Run prediction
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output[0], dim=0)

        # Decode the output
        classes = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
        predicted_class = classes[probabilities.argmax().item()]
        confidence = probabilities.max().item()

        # Run GradCAM
        gradcam = GradCAM(model, target_layer=model.features[-1])  # Target last convolutional layer
        class_idx = probabilities.argmax().item()
        heatmap = gradcam.generate_heatmap(input_tensor, target_class=class_idx)

        # Overlay heatmap on the image
        overlay_image = overlay_heatmap(image, heatmap)

        # Display images in columns (side-by-side)
        st.subheader("Prediction Results")
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)

        with col2:
            st.image(overlay_image, caption="GradCAM Heatmap", use_column_width=True)

        # Display prediction and confidence along with patient information
        st.write("### Patient Information")
        st.write(f"**Name:** {patient_name}")
        st.write(f"**Age:** {patient_age}")
        st.write(f"**ID:** {patient_id}")
        st.write(f"**MRI Test Date:** {mri_test_date}")
        st.write(f"**Gender:** {gender}")
        st.write(f"**Province:** {selected_province}")

        st.write("### Prediction Result")
        st.write(f"**Prediction:** {predicted_class}")
        st.write(f"**Confidence:** {confidence:.2f}")

        # Kontrol threshold
        threshold = st.slider("Activation Threshold", 0.1, 1.0, 0.5, 0.05)
        activation_coords = gradcam.get_activation_coordinates(heatmap, threshold)

        # Tampilkan hasil
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)
        with col2:
            overlay = overlay_heatmap(image, heatmap)
            st.image(overlay, caption="GradCAM Heatmap with Activation Points", use_column_width=True)

        # Analisis aktivasi
        st.subheader("Activation Analysis")
        if activation_coords:
            st.write(f"**Total Activated Pixels:** {len(activation_coords)}")

            col_meta1, col_meta2 = st.columns(2)
            with col_meta1:
                st.write("**Top 10 Activation Coordinates (X, Y):**")
                st.table(pd.DataFrame(activation_coords[:10], columns=["X", "Y"]))

            with col_meta2:
                st.write("**Heatmap Statistics:**")
                st.write(f"- Min Intensity: {np.min(heatmap):.2f}")
                st.write(f"- Max Intensity: {np.max(heatmap):.2f}")
                st.write(f"- Mean Intensity: {np.mean(heatmap):.2f}")

            # Download coordinates
            df_coords = pd.DataFrame(activation_coords, columns=["X", "Y"])
            csv = df_coords.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Full Coordinates (CSV)",
                data=csv,
                file_name="activation_coordinates.csv",
                mime="text/csv"
            )
        else:
            st.warning("No significant activations detected above threshold!")

        # Health information table
        st.write("### Health Analysis")
        st.table({
            "Category": ["Patient Name", "Age", "MRI Test Date", "Gender", "Province", "Prediction", "Confidence"],
            "Details": [patient_name, patient_age, mri_test_date, gender, selected_province, predicted_class, f"{confidence:.2f}"]
        })

        # Explanation of Prediction
        st.write("### Explanation of Prediction")
        if predicted_class == "Glioma":
            st.write("""
            **Glioma**
            - **Description:** Tumor ganas yang berasal dari sel glial di otak atau tulang belakang, sering bersifat agresif dan berkembang cepat.
            - **Penanganan Awal:** 
              1. Konsultasi neurologis: Pasien segera dirujuk ke spesialis bedah saraf atau onkologi.
              2. Biopsi atau operasi diagnostik: Untuk memastikan jenis dan tingkat keparahan tumor.
              3. Perencanaan terapi: Mulai perencanaan untuk operasi pengangkatan tumor (jika memungkinkan), radioterapi, atau kemoterapi.
              4. Pengelolaan gejala: Diberikan obat seperti steroid untuk mengurangi pembengkakan otak atau antiepilepsi untuk mencegah kejang.
            """)
        elif predicted_class == "Meningioma":
            st.write("""
            **Meningioma**
            - **Description:** Tumor yang berasal dari meninges (lapisan pelindung otak). Mayoritas bersifat jinak, tetapi bisa menekan struktur otak dan menyebabkan gejala jika ukurannya besar.
            - **Penanganan Awal:** 
              1. Pemantauan: Jika tumor
              2. Operasi: Jika tumor besar atau menyebabkan gejala, operasi pengangkatan menjadi langkah utama.
              3. Radioterapi: Digunakan untuk sisa tumor yang tidak bisa diangkat melalui operasi atau untuk tumor yang tumbuh kembali.
            """)
        elif predicted_class == "No Tumor":
            st.write("""
            **No Tumor (Tidak Ada Tumor)**
            - **Description:** MRI tidak menunjukkan adanya tumor di otak.
            - **Penanganan Awal:** 
              1. Identifikasi gejala lain: Jika pasien memiliki gejala neurologis, dokter akan mencari penyebab lain seperti infeksi, gangguan saraf, atau kondisi sistemik.
              2. Pemantauan rutin: Jika pasien berisiko tinggi, tetap dianjurkan untuk kontrol rutin.
              3. Pengobatan gejala: Misalnya, pemberian obat sakit kepala atau kejang jika ada gejala.
            """)
        elif predicted_class == "Pituitary":
            st.write("""
            **Pituitary Tumor**
            - **Description:** Tumor di kelenjar pituitari (hipofisis), sering jinak tetapi dapat memengaruhi produksi hormon.
            - **Penanganan Awal:** 
              1. Tes hormon: Dilakukan untuk memeriksa fungsi hormon dan menentukan apakah tumor aktif secara hormon (menghasilkan hormon berlebih).
              2. Pemeriksaan visual: Tumor pituitari besar dapat menekan saraf optik, sehingga pemeriksaan penglihatan dilakukan.
              3. Obat-obatan: Untuk tumor yang aktif secara hormon, terapi obat seperti agonis dopamin mungkin diresepkan.
              4. Operasi: Tumor besar atau yang memengaruhi fungsi otak dan saraf biasanya diangkat melalui prosedur bedah.
              5. Radioterapi: Jika operasi tidak memungkinkan atau tidak sepenuhnya berhasil.
            """)
