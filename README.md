## Project Description
This project applies an ensemble learning approach using EfficientNetV2, Xception, and ResNet50 to classify brain tumors based on MRI images, enhancing accuracy and robustness in detecting tumor types for medical diagnostics and research. To improve interpretability, Grad-CAM (Gradient-weighted Class Activation Mapping) is integrated, allowing visualization of important regions in MRI images that influence the model’s predictions, helping medical professionals understand and validate AI-based decisions.
![Demo](BrainTumorClasification.png)  

## Key Features:
- Ensemble Learning combining EfficientNetV2, Xception, and ResNet50
- High-accuracy classification for brain tumor detection
- Custom Dataset Support for improved generalization
- Grad-CAM Visualization for model interpretability
- Web Deployment using Streamlit

# Brain Tumor Classification using MRI and Ensemble Learning  

## 📌 Overview  
This project implements **ensemble learning** with **EfficientNetV2, Xception, and ResNet50** for classifying brain tumors from MRI images. The goal is to improve accuracy and model robustness, aiding in medical diagnosis.  

## 🛠 Features  
- **Ensemble Model** combining EfficientNetV2, Xception, and ResNet50  
- **MRI-based Brain Tumor Classification**  
- **Grad-CAM Visualization** for explainability  
- **Web Streamlit real-time classification**  
- **Support for Custom MRI Datasets**  

## 🖥️ Installation  
### Prerequisites  
- Python 3.8+  
- TensorFlow/Keras  
- OpenCV  
- NumPy  
- Streamlit

📁 brain-tumor-classification  
│── 📁 dataset/               # Training and test dataset  
│── 📁 models/                # Pretrained ensemble models  
│── 📁 scripts/               # Training and visualization scripts  
│── classify.py               # Inference script  
│── train.py                  # Model training script  
│── gradcam.py                # Grad-CAM heatmap visualization  
│── app.py                    # API deployment  
│── requirements.txt           # Dependencies  
│── README.md                  # Project documentation  

### Setup  
1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/brain-tumor-classification.git
   cd brain-tumor-classification
