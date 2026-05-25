"""Simplified FastAPI application for Alzheimer's MRI classification with EfficientNet-B2."""
import io
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image
import torchvision.transforms as transforms
import cv2
import numpy as np

# Paths
BASE_DIR = Path(__file__).parent.absolute()
MODEL_PATH = BASE_DIR / "model" / "best_model_b2.pth"
FRONTEND_DIR = BASE_DIR / "frontend"

# Class metadata (matching frontend expectations)
CLASS_NAMES = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
CLASS_DESCRIPTIONS = {
    "MildDemented": "Mild cognitive impairment with noticeable changes in brain structure.",
    "ModerateDemented": "Significant cognitive impairment with substantial structural changes in the brain.",
    "NonDemented": "No significant cognitive impairment detected. Brain structure appears normal.",
    "VeryMildDemented": "Early signs of cognitive decline. Minimal changes in brain structure.",
}

# Device detection with MPS support for Apple Silicon
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# Global model variable
model = None


class EfficientNetB2Classifier(nn.Module):
    """EfficientNet-B2 classifier for Alzheimer's detection."""
    def __init__(self, num_classes=4):
        super().__init__()
        # Load pretrained EfficientNet-B2
        from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights

        # Get the base model structure
        base_model = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)

        # Extract features and classifier directly to match saved checkpoint structure
        self.features = base_model.features

        # Modify classifier for 4 classes
        in_features = base_model.classifier[1].in_features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def load_model():
    """Load the trained EfficientNet-B2 model."""
    global model
    try:
        print(f"Loading model from {MODEL_PATH}...")
        print(f"Using device: {device}")

        # Initialize model
        model = EfficientNetB2Classifier(num_classes=4)

        # Load checkpoint
        checkpoint = torch.load(MODEL_PATH, map_location=device)

        # Load state dict
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            val_acc = checkpoint.get('val_acc', 'N/A')
            print(f"Model loaded successfully! Validation accuracy: {val_acc}")
        else:
            model.load_state_dict(checkpoint)
            print("Model loaded successfully!")

        model.to(device)
        model.eval()

        print("✅ Model is ready for inference")
        return True
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False


class ActivationMap:
    """Activation Map visualization for EfficientNet-B2 - shows feature map activations."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.hook_handle = None

        # Register forward hook
        self.register_hook()

    def register_hook(self):
        """Register forward hook to capture activations."""
        def forward_hook(module, input, output):
            self.activations = output.detach().clone()

        # Find the target layer and register hook
        for name, module in self.model.features.named_modules():
            if name == self.target_layer:
                self.hook_handle = module.register_forward_hook(forward_hook)
                break

    def remove_hook(self):
        """Remove the registered hook."""
        if self.hook_handle:
            self.hook_handle.remove()
            self.hook_handle = None

    def generate_activation_map(self, input_tensor):
        """Generate activation map by aggregating feature maps."""
        try:
            # Forward pass to capture activations
            with torch.no_grad():
                _ = self.model(input_tensor)

            if self.activations is None:
                raise ValueError("Failed to capture activations")

            # Get activations [1, C, H, W]
            activations = self.activations

            # Aggregate feature maps by mean
            activation_map = activations.mean(dim=1, keepdim=True)  # [1, 1, H, W]

            # Apply ReLU to focus on positive activations
            activation_map = F.relu(activation_map)

            # Resize to 224x224
            activation_map = F.interpolate(activation_map, size=(224, 224), mode='bilinear', align_corners=False)

            # Convert to numpy
            activation_map = activation_map.squeeze().cpu().numpy()

            # Normalize to 0-255
            if activation_map.max() > 0:
                activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min())
            activation_map = (activation_map * 255).astype(np.uint8)

            return activation_map

        except Exception as e:
            print(f"Activation map generation error: {e}")
            raise

    def generate_overlay(self, image, activation_map):
        """Generate overlay of activation map on original image."""
        # Apply colormap
        heatmap = cv2.applyColorMap(activation_map, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Resize heatmap to match image
        if image.shape[:2] != heatmap.shape[:2]:
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

        # Blend with original image
        overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

        return overlay, heatmap


def generate_gradcam(image_tensor, original_image):
    """Generate attention visualization using activation maps."""
    try:
        print("Generating attention visualization...")
        # Create ActivationMap instance - use last conv layer for best results
        activation_map = ActivationMap(model, target_layer="8.1")  # Last conv block in EfficientNet-B2

        # Generate activation map
        attention_map = activation_map.generate_activation_map(image_tensor)

        # Convert original image to numpy
        original_np = np.array(original_image)

        # Handle grayscale
        if len(original_np.shape) == 2:
            original_np = cv2.cvtColor(original_np, cv2.COLOR_GRAY2RGB)

        # Ensure RGB
        if original_np.shape[2] == 4:
            original_np = original_np[:, :, :3]

        # Resize to 224x224 for processing
        original_resized = cv2.resize(original_np, (224, 224))

        # Generate overlays
        overlay, heatmap = activation_map.generate_overlay(original_resized, attention_map)

        # Convert to base64
        import base64
        from io import BytesIO

        def encode_image(img_array):
            """Convert numpy array to base64 string."""
            img = Image.fromarray(img_array.astype(np.uint8))
            buffer = BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')

        return {
            "overlay": encode_image(overlay),
            "heatmap": encode_image(heatmap)
        }
    except Exception as e:
        print(f"Attention visualization failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Create FastAPI app
app = FastAPI(
    title="NeuroScan - Alzheimer's MRI Classification",
    description="EfficientNet-B2 powered brain MRI classification for Alzheimer's Disease detection",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend
if FRONTEND_DIR.exists():
    app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")
    app.mount("/css", StaticFiles(directory=str(FRONTEND_DIR / "css")), name="css")
    app.mount("/js", StaticFiles(directory=str(FRONTEND_DIR / "js")), name="js")
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR)), name="assets")


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global model
    success = load_model()
    if not success:
        print("⚠️  Warning: Application started but model loading failed!")


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "device": str(device),
        "model_type": "EfficientNet-B2"
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload an MRI scan and receive a classification prediction.

    - **file**: MRI image file (JPEG or PNG)
    - **Returns**: Predicted class, confidence scores, and probabilities
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please check server logs.")

    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Only JPEG and PNG images are supported")

    # Read image
    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=400, detail="Image too large (max 10MB)")

    try:
        # Load and preprocess image
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        original_image = image.copy()  # Keep original for GradCAM
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Get prediction
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)[0].cpu().numpy()

        # Get predicted class
        predicted_idx = probabilities.argmax()
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = float(probabilities[predicted_idx])

        # Build probabilities dict
        probs_dict = {
            CLASS_NAMES[i]: float(probabilities[i])
            for i in range(len(CLASS_NAMES))
        }

        # Generate GradCAM attention heatmaps
        gradcam_images = None
        try:
            print("Generating GradCAM attention heatmaps...")
            gradcam_images = generate_gradcam(input_tensor, original_image)
            if gradcam_images:
                print("✅ GradCAM generated successfully")
            else:
                print("⚠️  GradCAM generation returned None")
        except Exception as gradcam_error:
            print(f"⚠️  GradCAM generation failed: {gradcam_error}")
            import traceback
            traceback.print_exc()
            gradcam_images = None

        # Build response
        response = {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "description": CLASS_DESCRIPTIONS[predicted_class],
            "probabilities": probs_dict,
            "model_details": {
                "model_architecture": "EfficientNet-B2",
                "input_resolution": "224x224",
                "accuracy": "99.39%",
                "device": str(device)
            }
        }

        # Add GradCAM images if generation succeeded
        if gradcam_images:
            response["gradcam_images"] = gradcam_images

        return response

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/")
async def root():
    """Serve frontend."""
    frontend_path = FRONTEND_DIR / "index.html"
    if frontend_path.exists():
        return FileResponse(frontend_path)
    return {"message": "NeuroScan - Alzheimer's MRI Classification API", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)