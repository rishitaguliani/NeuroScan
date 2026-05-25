"""Simplified FastAPI application for Alzheimer's MRI classification with EfficientNet-B2."""
import io
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image
import torchvision.transforms as transforms

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
        self.backbone = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)

        # Modify classifier for 4 classes
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


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

        # Build response
        return {
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