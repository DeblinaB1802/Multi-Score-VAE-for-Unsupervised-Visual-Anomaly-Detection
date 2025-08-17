# Multi-Score VAE for Visual Anomaly Detection

A Variational Autoencoder (VAE) implementation that combines reconstruction loss and latent-space Mahalanobis distance for robust visual anomaly detection, with explainability features through heatmap visualizations.

## 🔍 Overview

This project implements a novel approach to visual anomaly detection by leveraging a multi-score system that combines:
- **Reconstruction Loss**: Measures how well the VAE can reconstruct input images
- **Latent-Space Mahalanobis Distance**: Captures statistical deviations in the learned latent representation

The model provides explainable results through heatmaps that highlight anomalous regions using reconstruction differences and Grad-CAM visualizations.

## 📊 Performance

Evaluated on Oxford-102 Flowers dataset (In-Distribution) vs CIFAR-10 (Out-of-Distribution):

| Metric | Score |
|--------|-------|
| **Accuracy** | 84.7% |
| **AUROC** | 0.81 |
| **F1-Score** | 0.79 |

## 🏗️ Architecture

### Variational Autoencoder Components
- **Encoder**: Maps input images to latent distribution parameters (μ, σ)
- **Decoder**: Reconstructs images from latent samples
- **Latent Space**: Learns meaningful representations for anomaly detection

### Multi-Score System
1. **Reconstruction Score**: L2 distance between input and reconstructed image
2. **Mahalanobis Score**: Statistical distance in latent space using learned covariance
3. **Combined Score**: Weighted combination of both metrics

## 🚀 Features

- ✅ **Dual-metric anomaly detection** for improved robustness
- ✅ **Explainable AI** with reconstruction difference heatmaps
- ✅ **Grad-CAM integration** for attention visualization
- ✅ **Comprehensive evaluation** on standard benchmarks
- ✅ **Modular design** for easy experimentation

## 📋 Requirements

```
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
matplotlib>=3.3.0
scikit-learn>=0.24.0
opencv-python>=4.5.0
pillow>=8.0.0
tqdm>=4.62.0
```

## 🛠️ Installation

1. Clone or download the repository:
```bash
git clone https://github.com/yourusername/multi-score-vae-anomaly-detection.git
cd multi-score-vae-anomaly-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook Multi_Score_VAE_Anomaly_Detection.ipynb
```

## 📁 Project Structure

```
multi-score-vae-anomaly-detection/
├── Multi_Score_VAE_Anomaly_Detection.ipynb    # Main implementation notebook
├── requirements.txt                            # Dependencies
└── README.md                                  # This file
```

### Notebook Contents
The Jupyter notebook is organized into the following sections:
- **Data Loading & Preprocessing**: Oxford-102 Flowers and CIFAR-10 setup
- **VAE Architecture**: Encoder, decoder, and latent space implementation
- **Multi-Score System**: Reconstruction + Mahalanobis distance scoring
- **Training Loop**: Model training with β-VAE loss
- **Evaluation**: Performance metrics and threshold optimization
- **Explainability**: Heatmap generation and Grad-CAM visualization
- **Results Visualization**: Performance plots and anomaly examples

## 🎯 Usage

### Running the Notebook

1. **Open the notebook**: Launch `Multi_Score_VAE_Anomaly_Detection.ipynb` in Jupyter
2. **Install dependencies**: Run the first cell to install required packages
3. **Execute cells sequentially**: Follow the notebook from data loading through evaluation
4. **Customize parameters**: Modify hyperparameters in the configuration cells
5. **View results**: Check generated plots, metrics, and heatmap visualizations

### Key Notebook Sections

**Data Preparation**
```python
# Download and prepare Oxford-102 Flowers (ID) and CIFAR-10 (OOD) datasets
# Apply transforms and create data loaders
```

**Model Definition**
```python
# Define VAE architecture with encoder/decoder
# Implement multi-score anomaly detection system
```

**Training**
```python
# Train VAE with β-VAE loss
# Monitor reconstruction and KL divergence losses
```

**Evaluation & Explainability**
```python
# Calculate anomaly scores using dual metrics
# Generate reconstruction heatmaps and Grad-CAM visualizations
# Compute final performance metrics
```

## 📖 Methodology

### 1. Data Preparation
- **In-Distribution**: Oxford-102 Flowers dataset (8,189 images, 102 flower categories)
- **Out-of-Distribution**: CIFAR-10 dataset (10,000 test images, 10 object categories)
- **Preprocessing**: Resize to 224×224, normalize to [0,1], data augmentation during training

### 2. Model Training
- **Loss Function**: β-VAE loss with KL divergence regularization
- **Optimization**: Adam optimizer with learning rate scheduling
- **Regularization**: Dropout, batch normalization, early stopping

### 3. Anomaly Scoring
- **Reconstruction Score**: `||x - x̂||₂²`
- **Mahalanobis Score**: `(z - μ)ᵀ Σ⁻¹ (z - μ)`
- **Combined Score**: `α × reconstruction_score + (1-α) × mahalanobis_score`

### 4. Explainability
- **Reconstruction Heatmaps**: Pixel-wise difference between input and reconstruction
- **Grad-CAM**: Gradient-based attention maps highlighting important regions

## 📈 Results Analysis

### Quantitative Results
The multi-score approach significantly outperforms single-metric baselines:
- **Reconstruction-only**: AUROC 0.76, F1 0.72
- **Mahalanobis-only**: AUROC 0.78, F1 0.75
- **Multi-score (ours)**: AUROC 0.81, F1 0.79

### Qualitative Analysis
- Reconstruction heatmaps effectively highlight texture and structural anomalies
- Grad-CAM visualizations reveal model attention on discriminative features
- Combined approach reduces false positives compared to individual metrics

## 🔧 Configuration

Key hyperparameters configured within the notebook:

```python
# Model Configuration
LATENT_DIM = 128
BETA = 1.0  # β-VAE regularization weight
HIDDEN_DIMS = [512, 256, 128]

# Training Configuration
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 100
WEIGHT_DECAY = 1e-4

# Anomaly Detection Configuration
ALPHA = 0.6  # Weight for reconstruction vs Mahalanobis score
THRESHOLD = 0.5  # Anomaly detection threshold
```

## 🧪 Running the Experiment

To reproduce the results:

1. **Open the notebook**: `Multi_Score_VAE_Anomaly_Detection.ipynb`
2. **Run all cells sequentially** to:
   - Download and preprocess datasets
   - Define and train the VAE model
   - Evaluate anomaly detection performance
   - Generate explainability visualizations
3. **Experiment with parameters** by modifying configuration cells
4. **Save results** using the built-in visualization and export functions

The notebook automatically handles:
- Dataset downloading and preprocessing
- Model training with progress monitoring
- Performance evaluation and metric calculation
- Heatmap and Grad-CAM generation
- Results visualization and saving

## 📚 References

1. Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114.
2. Lee, K., et al. (2018). A simple unified framework for detecting out-of-distribution samples and adversarial attacks. NeurIPS.
3. Selvaraju, R. R., et al. (2017). Grad-cam: Visual explanations from deep networks via gradient-based localization. ICCV.
4. Nilsback, M. E., & Zisserman, A. (2008). Automated flower classification over a large number of classes. ICVGIP.

## 🙏 Acknowledgments

- Oxford Visual Geometry Group for the Oxford-102 Flowers dataset
- CIFAR team for the CIFAR-10 dataset
- PyTorch community for the deep learning framework

---

**Note**: This project was developed as part of a research initiative in visual anomaly detection (Feb'25–Mar'25). For questions or collaboration opportunities, please reach out via the contact information above.
