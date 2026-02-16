import argparse
import os
import sys

sys.path.append(os.getcwd())

from src.components.inference import InferenceModel

def main():
    parser = argparse.ArgumentParser(description="Vehicle Damage Assessment Inference")

    parser.add_argument(
        "--image", 
        type=str, 
        required=True, 
        help="Ruta a la imagen del coche que quieres analizar"
    )


    parser.add_argument("--parts_model", type=str, default="C:\\Users\\angel\\Desktop\\Proyecto\\vehicle-damage-assesment\\models\\car_parts_model.pt", help="Ruta al modelo de Partes")
    parser.add_argument("--damage_model", type=str, default="C:\\Users\\angel\\Desktop\\Proyecto\\vehicle-damage-assesment\\models\\car_damages_model.pt", help="Ruta al modelo de Daños")
    parser.add_argument("--output_dir", type=str, default="C:\\Users\\angel\\Desktop\\Proyecto\\vehicle-damage-assesment\\results", help="Carpeta donde guardar las imágenes procesadas")

    args = parser.parse_args()

    # 2. Verificaciones
    if not os.path.exists(args.image):
        print(f"❌ Error: La imagen '{args.image}' no existe.")
        return

    # 3. Instanciar el Sistema
    try:
        assessor = InferenceModel(parts_model_path=args.parts_model, damage_model_path=args.damage_model)
    except Exception as e:
        print(f"❌ Error crítico cargando modelos. Revisa las rutas. Detalles: {e}")
        return

    # 4. Ejecutar la Predicción y Visualización
    print("-" * 50)
    # AHORA RECIBIMOS DOS COSAS: El reporte y la ruta de la imagen guardada
    reporte_texto, ruta_imagen = assessor.predict_and_visualize(args.image, args.output_dir)
    print("-" * 50)
    
    # 5. Mostrar Resultados en Terminal
    if not reporte_texto:
        print("\n✅ No se han detectado daños relevantes (o no se pudieron localizar).")
    else:
        print("\n📋 REPORTE FINAL DE PERITACIÓN")
        print("=" * 30)
        for pieza, lista_daños in reporte_texto.items():
            print(f"🔴 ZONA: {pieza.upper()}")
            for daño in lista_daños:
                print(f"   └── ⚠️ {daño}")
            print("-" * 30)
    
    print(f"\n🖼️ Puedes ver el resultado visual en: {ruta_imagen}")
    print("=" * 50)

if __name__ == "__main__":
    main()