import cv2
import numpy as np
from ultralytics import YOLO
import os

class InferenceModel:
    def __init__(self, parts_model_path, damage_model_path):
        print("🏗️ Cargando modelos de IA...")
        self.parts_model = YOLO(parts_model_path)
        self.damage_model = YOLO(damage_model_path)
        print("✅ Modelos cargados.")

    def _get_center(self, box):
        x1, y1, x2, y2 = box
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def predict_and_visualize(self, image_path, output_dir="outputs"):
        print(f"📸 Analizando: {os.path.basename(image_path)}")
        
        # Cargar imagen original
        img_cv2 = cv2.imread(image_path)
        if img_cv2 is None:
            raise ValueError("No se pudo leer la imagen.")
        h_img, w_img, _ = img_cv2.shape

        # --- 1. MODELO DE PARTES (Imagen Completa) ---
        res_parts = self.parts_model.predict(source=img_cv2, imgsz=1024, conf=0.5, verbose=False)[0]
        
        # Preparamos la imagen base
        vis_img = res_parts.plot(line_width=2, boxes=False)

        # --- 2. MODELO DE DAÑOS (Tiling + NMS) ---
        print("🔍 Escaneando daños por sectores (Tiling)...")
        
        overlap = 0.1 # 10%
        mid_x, mid_y = int(w_img / 2), int(h_img / 2)
        
        crops_coords = [
            (0, 0, mid_x + int(mid_x*overlap), mid_y + int(mid_y*overlap)),
            (mid_x - int(mid_x*overlap), 0, w_img, mid_y + int(mid_y*overlap)),
            (0, mid_y - int(mid_y*overlap), mid_x + int(mid_x*overlap), h_img),
            (mid_x - int(mid_x*overlap), mid_y - int(mid_y*overlap), w_img, h_img)
        ]

        # Listas temporales para el algoritmo NMS
        raw_boxes_nms = []  # Formato [x, y, w, h] necesario para cv2.dnn.NMSBoxes
        raw_scores = []     # Confianza de cada detección
        raw_details = []    # Datos extra (nombre, caja global exacta, clase)

        # Procesamos cada recorte
        for (x1, y1, x2, y2) in crops_coords:
            crop_img = img_cv2[y1:y2, x1:x2]
            
            # Inferencia en el recorte
            res_crop = self.damage_model.predict(source=crop_img, imgsz=640, conf=0.2, verbose=False)[0]

            if res_crop.boxes is not None:
                for i, box in enumerate(res_crop.boxes.xyxy.cpu().numpy()):
                    # Filtro de nombre
                    cls_id = int(res_crop.boxes.cls[i])
                    raw_name = res_crop.names[cls_id]
                    if "no damage" in raw_name.lower():
                        continue

                    # Coordenadas locales
                    bx1, by1, bx2, by2 = box
                    conf = float(res_crop.boxes.conf[i])

                    # Transformar a Coordenadas Globales
                    gx1 = int(bx1 + x1)
                    gy1 = int(by1 + y1)
                    gx2 = int(bx2 + x1)
                    gy2 = int(by2 + y1)

                    # Preparar datos para NMS (x, y, ancho, alto)
                    w_box = gx2 - gx1
                    h_box = gy2 - gy1
                    
                    raw_boxes_nms.append([gx1, gy1, w_box, h_box])
                    raw_scores.append(conf)
                    
                    # Guardamos los detalles para recuperarlos después del filtrado
                    raw_details.append({
                        "global_box": [gx1, gy1, gx2, gy2],
                        "name": raw_name,
                        "cls_id": cls_id
                    })

        # --- 3. APLICAR FILTRO DE DUPLICADOS (NMS) ---
        # Si dos cajas se solapan más del 30% (nms_threshold=0.3), borramos la peor
        final_indices = []
        if len(raw_boxes_nms) > 0:
            indices = cv2.dnn.NMSBoxes(raw_boxes_nms, raw_scores, score_threshold=0.2, nms_threshold=0.3)
            # cv2.dnn.NMSBoxes devuelve una tupla o lista dependiendo de la versión, aplanamos:
            if len(indices) > 0:
                final_indices = indices.flatten()
        
        print(f"🧹 Detecciones iniciales: {len(raw_boxes_nms)} -> Tras limpieza: {len(final_indices)}")

        # --- 4. PINTAR Y REPORTE (Solo las cajas supervivientes) ---
        report = {}
        
        for idx in final_indices:
            data = raw_details[idx]
            gx1, gy1, gx2, gy2 = data["global_box"]
            d_name = data["name"]

            # Pintamos rectángulo
            cv2.rectangle(vis_img, (gx1, gy1), (gx2, gy2), (0, 0, 255), 3)
            
            # Poner etiqueta
            cv2.putText(vis_img, d_name, (gx1, gy1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

            # Lógica de Cruce con Partes
            cx, cy = self._get_center([gx1, gy1, gx2, gy2])
            piece_found = "Zona Desconocida"

            if res_parts.masks is not None:
                for j, mask_poly in enumerate(res_parts.masks.xy):
                    if len(mask_poly) == 0: continue
                    is_inside = cv2.pointPolygonTest(np.array(mask_poly, dtype=np.int32), (cx, cy), False)
                    if is_inside >= 0:
                        cls_id_part = int(res_parts.boxes.cls[j])
                        piece_found = res_parts.names[cls_id_part]
                        break

            if piece_found not in report:
                report[piece_found] = []
            report[piece_found].append(d_name)

        # --- 5. GUARDAR IMAGEN ---
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(image_path)
        save_path = os.path.join(output_dir, f"result_{filename}")
        cv2.imwrite(save_path, vis_img)
        print(f"💾 Imagen guardada en: {save_path}")

        return report, save_path