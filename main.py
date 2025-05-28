from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
import torch
from torchvision import transforms
from PIL import Image
from model import Vit
import os
import io

app = FastAPI()

@app.middleware("http")
async def add_head_support(request: Request, call_next):
    if request.method == "HEAD":
        request.scope["method"] = "GET"
        response = await call_next(request)
        response.body = b""
        return response
    else:
        response = await call_next(request)
        return response

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Vit().to(device)

pth_path = "brain-tumour.pth"
if os.path.exists(pth_path):
    checkpoint = torch.load(pth_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
else:
    raise RuntimeError("Model .pth file not found!")


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])

label_map = {
    0: "Glioma",
    1: "Meningioma",
    2: "No tumor",
    3: "Pituitary"
}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("L")
        image = transform(image).unsqueeze(0).to(device)

        with torch.inference_mode():
            output = model(image)
            _, pred = torch.max(output, 1)
            label = label_map.get(pred.item(), "Unknown")

        return JSONResponse(content={"prediction": label})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
from fastapi import Response

@app.get("/predict", include_in_schema=False)
@app.get("/predict/", include_in_schema=False)
def ping_predict():
    return {"message": "Ping OK"}


@app.get("/")
def root():
    return {"message": "Upload an image to /predict/ to get the tumor class"}
