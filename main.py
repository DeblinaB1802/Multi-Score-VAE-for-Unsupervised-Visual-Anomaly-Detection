# Multi-Score VAE for Visual Anomaly Detection
# Combines Reconstruction Loss & Latent-Space Mahalanobis Distance

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from tqdm import tqdm
import cv2

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Configuration
CONFIG = {
    'latent_dim': 128,
    'hidden_dims': [512, 256, 128],
    'beta': 1.0,
    'batch_size': 64,
    'learning_rate': 1e-3,
    'epochs': 50,
    'alpha': 0.6,  # Weight for combining scores
    'image_size': 64,
    'channels': 3
}

# Data Loading and Preprocessing
def get_data_loaders():
    """Load Oxford Flowers (ID) and CIFAR-10 (OOD) datasets"""
    transform = transforms.Compose([
        transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Oxford Flowers (In-Distribution) - using subset for demo
    try:
        flowers_data = torchvision.datasets.Flowers102(
            root='./data', split='train', download=True, transform=transform
        )
    except:
        # Fallback to CIFAR-100 if Flowers102 unavailable
        flowers_data = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform
        )
        flowers_data.data = flowers_data.data[:5000]  # Use subset
    
    # CIFAR-10 (Out-of-Distribution)
    cifar_data = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    flowers_loader = DataLoader(flowers_data, batch_size=CONFIG['batch_size'], shuffle=True)
    cifar_loader = DataLoader(cifar_data, batch_size=CONFIG['batch_size'], shuffle=False)
    
    return flowers_loader, cifar_loader

# VAE Architecture
class VAE(nn.Module):
    def __init__(self, latent_dim=128, hidden_dims=[512, 256, 128]):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate flattened size
        self.feature_size = 256 * (CONFIG['image_size'] // 16) ** 2
        
        self.fc_mu = nn.Linear(self.feature_size, latent_dim)
        self.fc_logvar = nn.Linear(self.feature_size, latent_dim)
        
        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, self.feature_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Tanh()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.decoder_fc(z)
        h = h.view(-1, 256, CONFIG['image_size'] // 16, CONFIG['image_size'] // 16)
        return self.decoder(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z

# Loss Function
def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """β-VAE loss with reconstruction and KL divergence"""
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss

# Multi-Score Anomaly Detector
class MultiScoreDetector:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.mu_train = None
        self.cov_inv = None
    
    def fit_mahalanobis(self, train_loader):
        """Fit Mahalanobis parameters on training data"""
        latents = []
        with torch.no_grad():
            for batch, _ in tqdm(train_loader, desc="Fitting Mahalanobis"):
                batch = batch.to(device)
                mu, _ = self.model.encode(batch)
                latents.append(mu.cpu())
        
        latents = torch.cat(latents, dim=0)
        self.mu_train = latents.mean(dim=0)
        cov = torch.cov(latents.T)
        self.cov_inv = torch.linalg.pinv(cov)
    
    def compute_scores(self, x):
        """Compute reconstruction and Mahalanobis scores"""
        with torch.no_grad():
            recon, mu, _, _ = self.model(x)
            
            # Reconstruction score
            recon_score = F.mse_loss(recon, x, reduction='none').view(x.size(0), -1).mean(dim=1)
            
            # Mahalanobis score
            if self.mu_train is not None:
                diff = mu.cpu() - self.mu_train
                mahal_score = torch.sum(diff @ self.cov_inv * diff, dim=1)
            else:
                mahal_score = torch.zeros(x.size(0))
            
            return recon_score.cpu(), mahal_score

# Training Function
def train_vae(model, train_loader, epochs=50):
    """Train the VAE model"""
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    model.train()
    
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for batch, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch = batch.to(device)
            
            optimizer.zero_grad()
            recon, mu, logvar, _ = model(batch)
            loss, recon_loss, kl_loss = vae_loss(recon, batch, mu, logvar, CONFIG['beta'])
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    
    return losses

# Evaluation Function
def evaluate_anomaly_detection(detector, id_loader, ood_loader):
    """Evaluate anomaly detection performance"""
    scores_id, scores_ood = [], []
    
    # In-distribution scores
    for batch, _ in tqdm(id_loader, desc="Evaluating ID"):
        batch = batch.to(device)
        recon_score, mahal_score = detector.compute_scores(batch)
        combined_score = CONFIG['alpha'] * recon_score + (1 - CONFIG['alpha']) * mahal_score
        scores_id.extend(combined_score.numpy())
    
    # Out-of-distribution scores
    for batch, _ in tqdm(ood_loader, desc="Evaluating OOD"):
        batch = batch.to(device)
        recon_score, mahal_score = detector.compute_scores(batch)
        combined_score = CONFIG['alpha'] * recon_score + (1 - CONFIG['alpha']) * mahal_score
        scores_ood.extend(combined_score.numpy())
    
    # Create labels (0: normal, 1: anomaly)
    y_true = [0] * len(scores_id) + [1] * len(scores_ood)
    y_scores = scores_id + scores_ood
    
    # Calculate metrics
    threshold = np.percentile(scores_id, 95)  # 95th percentile of normal scores
    y_pred = [1 if score > threshold else 0 for score in y_scores]
    
    auroc = roc_auc_score(y_true, y_scores)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    return auroc, accuracy, f1, threshold

# Explainability Functions
def generate_reconstruction_heatmap(model, image):
    """Generate heatmap showing reconstruction differences"""
    model.eval()
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)
        recon, _, _, _ = model(image)
        
        # Compute absolute difference
        diff = torch.abs(image - recon).squeeze().cpu()
        diff = diff.mean(dim=0)  # Average across channels
        
        return diff.numpy()

def grad_cam(model, image, target_layer='encoder.4'):
    """Simple Grad-CAM implementation"""
    model.eval()
    image = image.unsqueeze(0).to(device)
    image.requires_grad_()
    
    # Forward pass
    recon, mu, logvar, _ = model(image)
    loss, _, _ = vae_loss(recon, image, mu, logvar)
    
    # Backward pass
    model.zero_grad()
    loss.backward(retain_graph=True)
    
    # Get gradients and activations
    gradients = image.grad.squeeze().cpu()
    activation_map = torch.abs(gradients).mean(dim=0)
    
    return activation_map.numpy()

# Visualization Functions
def plot_results(losses, auroc, accuracy, f1):
    """Plot training losses and results"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Training loss
    ax1.plot(losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    
    # Metrics bar plot
    metrics = ['AUROC', 'Accuracy', 'F1-Score']
    values = [auroc, accuracy, f1]
    ax2.bar(metrics, values)
    ax2.set_title('Performance Metrics')
    ax2.set_ylim(0, 1)
    
    # Add text annotations
    for i, v in enumerate(values):
        ax2.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    ax3.axis('off')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_anomalies(model, id_loader, ood_loader, n_samples=4):
    """Visualize normal and anomalous samples with heatmaps"""
    model.eval()
    
    fig, axes = plt.subplots(3, n_samples * 2, figsize=(16, 8))
    
    # Get samples
    id_batch = next(iter(id_loader))[0][:n_samples]
    ood_batch = next(iter(ood_loader))[0][:n_samples]
    
    all_samples = torch.cat([id_batch, ood_batch])
    labels = ['Normal'] * n_samples + ['Anomaly'] * n_samples
    
    for i, (img, label) in enumerate(zip(all_samples, labels)):
        # Original image
        axes[0, i].imshow(img.permute(1, 2, 0) * 0.5 + 0.5)
        axes[0, i].set_title(f'{label}')
        axes[0, i].axis('off')
        
        # Reconstruction
        with torch.no_grad():
            recon, _, _, _ = model(img.unsqueeze(0).to(device))
        recon_img = recon.squeeze().cpu()
        axes[1, i].imshow(recon_img.permute(1, 2, 0) * 0.5 + 0.5)
        axes[1, i].set_title('Reconstruction')
        axes[1, i].axis('off')
        
        # Heatmap
        heatmap = generate_reconstruction_heatmap(model, img)
        axes[2, i].imshow(heatmap, cmap='hot')
        axes[2, i].set_title('Anomaly Heatmap')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Main Execution
if __name__ == "__main__":
    print("Loading datasets...")
    train_loader, test_loader = get_data_loaders()
    
    print("Initializing model...")
    model = VAE(CONFIG['latent_dim'], CONFIG['hidden_dims']).to(device)
    
    print("Training VAE...")
    losses = train_vae(model, train_loader, CONFIG['epochs'])
    
    print("Setting up anomaly detector...")
    detector = MultiScoreDetector(model)
    detector.fit_mahalanobis(train_loader)
    
    print("Evaluating performance...")
    auroc, accuracy, f1, threshold = evaluate_anomaly_detection(detector, train_loader, test_loader)
    
    print(f"\nResults:")
    print(f"AUROC: {auroc:.3f}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"F1-Score: {f1:.3f}")
    
    print("Generating visualizations...")
    plot_results(losses, auroc, accuracy, f1)
    visualize_anomalies(model, train_loader, test_loader)
    
    print("Done!")
