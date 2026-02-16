# 1. Usamos una imagen base de Python ligera (Linux)
FROM python:3.10-slim

# 2. Instalamos librerías de sistema (CORREGIDO)
# En las nuevas versiones de Debian, 'libgl1-mesa-glx' se llama ahora 'libgl1'
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. Establecemos el directorio de trabajo dentro del contenedor
WORKDIR /app

# 4. Copiamos el archivo de requisitos e instalamos dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiamos TODO tu código al contenedor
COPY . .

# 6. Exponemos los puertos (8000 para API, 8501 para Streamlit)
EXPOSE 8000
EXPOSE 8501

# 7. Comando por defecto
CMD ["python", "app.py"]