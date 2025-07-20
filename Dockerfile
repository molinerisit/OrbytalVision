# Usa una imagen base de Python delgada pero funcional
FROM python:3.11-slim

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Instalar dependencias del sistema operativo que OpenCV necesita para funcionar
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copiar el archivo de requerimientos primero
COPY requirements.txt .

# Instalar las librerías de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el resto del código de tu proyecto al contenedor
COPY . .

# Exponer el puerto que nuestra app Flask usa
EXPOSE 8080

# El comando que se ejecutará para iniciar la aplicación
CMD ["python", "app.py"]