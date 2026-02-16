import os
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from src.components.inference import InferenceModel

# 1. Crear la "Instancia" de la aplicación
app = FastAPI(
    title="Vehicle Damage AI API",
    description="API para detectar daños en vehículos usando YOLOv8 (Segmentación + Tiling)",
    version="1.0.0"
)

# 2. Cargar los Modelos AL INICIAR (Para no cargarlos en cada petición)
# Ajusta las rutas si son diferentes en tu ordenador
PARTS_MODEL = "models/car_parts_model.pt"
DAMAGE_MODEL = "models/car_damages_model.pt"

# Instanciamos tu clase (El Cocinero)
print("⏳ Iniciando servidor y cargando modelos...")
try:
    assessor = InferenceModel(PARTS_MODEL, DAMAGE_MODEL)
    print("✅ API lista para recibir imágenes.")
except Exception as e:
    print(f"❌ Error cargando modelos: {e}")

# 3. Definir el "Endpoint" (La ventanilla de pedidos)
# POST: Porque el usuario nos "envía" datos (la foto)
@app.post("/predict")
async def predict_damage(file: UploadFile = File(...)):
    """
    Sube una imagen de un coche y recibe el reporte de daños y la imagen procesada.
    """
    
    # A. Guardar la imagen que nos envían en un archivo temporal
    # (Tu clase DamageAssessor necesita una ruta de archivo, no bytes en memoria)
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # B. Llamar a tu Inteligencia Artificial
        # Definimos una carpeta estática para guardar resultados
        output_folder = "api_results"
        os.makedirs(output_folder, exist_ok=True)
        
        # Ejecutamos la predicción
        reporte, ruta_imagen_guardada = assessor.predict_and_visualize(temp_filename, output_folder)

        # C. Devolver la respuesta (JSON)
        # Devolvemos el reporte y la URL (ruta) de la imagen generada
        return {
            "status": "success",
            "filename": file.filename,
            "reporte_daños": reporte,
            "imagen_procesada": ruta_imagen_guardada
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
    
    finally:
        # D. Limpieza: Borrar la imagen temporal original para no llenar el disco
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

# 4. Un Endpoint extra para descargar la imagen procesada (Opcional pero útil)
@app.get("/get-image")
async def get_image(image_path: str):
    if os.path.exists(image_path):
        return FileResponse(image_path)
    return {"error": "Image not found"}


if __name__ == "__main__":
    import uvicorn
    # Ejecuta el servidor en el puerto 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)