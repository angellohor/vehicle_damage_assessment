import os
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from src.components.inference import InferenceModel


app = FastAPI(
    title="Vehicle Damage AI API",
    description="API para detectar daños en vehículos usando YOLOv8 (Segmentación + Tiling)",
    version="1.0.0"
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


PARTS_MODEL = os.path.join(BASE_DIR, "models", "car_parts_model.pt")
DAMAGE_MODEL = os.path.join(BASE_DIR, "models", "car_damages_model.pt")


print("⏳ Iniciando servidor y cargando modelos...")
try:
    assessor = InferenceModel(PARTS_MODEL, DAMAGE_MODEL)
    print("✅ API lista para recibir imágenes.")
except Exception as e:
    print(f"❌ Error cargando modelos: {e}")



@app.post("/predict")
async def predict_damage(file: UploadFile = File(...)):
    
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        
        output_folder = "api_results"
        os.makedirs(output_folder, exist_ok=True)
        
       
        reporte, ruta_imagen_guardada = assessor.predict_and_visualize(temp_filename, output_folder)

        
        return {
            "status": "success",
            "filename": file.filename,
            "reporte_daños": reporte,
            "imagen_procesada": ruta_imagen_guardada
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
    
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

@app.get("/get-image")
async def get_image(image_path: str):
    if os.path.exists(image_path):
        return FileResponse(image_path)
    return {"error": "Image not found"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)