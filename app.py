import cv2
import numpy as np
import os
import time
import threading
import face_recognition
import shutil
from functools import wraps
from collections import deque
from datetime import datetime, timedelta
from flask import Flask, render_template, Response, request, jsonify, send_from_directory
from pymongo import MongoClient
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env (para desarrollo local)
load_dotenv()

# --- CONFIGURACIÓN (Leída desde Variables de Entorno) ---

# --- Variables Sensibles (Secretos) ---
MONGO_CONNECTION_STRING = os.environ.get("MONGO_URL")
RTSP_URL = os.environ.get("RTSP_URL")
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD")

# --- Comprobaciones de Seguridad al Inicio ---
if not MONGO_CONNECTION_STRING: raise ValueError("Error Crítico: La variable de entorno MONGO_URL no está definida.")
if not RTSP_URL: raise ValueError("Error Crítico: La variable de entorno RTSP_URL no está definida.")
if not ADMIN_PASSWORD: raise ValueError("Error Crítico: La variable de entorno ADMIN_PASSWORD no está definida.")

# --- Variables de Comportamiento (No Sensibles) ---
DETECTION_INTERVAL = int(os.environ.get("DETECTION_INTERVAL", "25"))
FACE_PADDING = float(os.environ.get("FACE_PADDING", "0.25"))
LOITERING_TIME_SECONDS = int(os.environ.get("LOITERING_TIME_SECONDS", "15"))
LOITERING_TOLERANCE_PIXELS = int(os.environ.get("LOITERING_TOLERANCE_PIXELS", "30"))

# --- Variables de Rutas (Paths) ---
SAVE_DIR = "capturas_faciales"
WATCHLIST_DIR = "watchlists"
ZONA_RESTRINGIDA = np.array([[10, 10], [400, 10], [400, 300], [10, 300]], np.int32)

# Crear carpetas necesarias
os.makedirs(SAVE_DIR, exist_ok=True)
for category in ['empleados', 'ladrones_conocidos', 'vips']:
    os.makedirs(os.path.join(WATCHLIST_DIR, category), exist_ok=True)

# --- CONFIGURACIÓN DE MONGODB ---
try:
    client = MongoClient(MONGO_CONNECTION_STRING, serverSelectionTimeoutMS=5000)
    client.server_info()
    db = client['vigilancia_db']
    eventos_collection = db['eventos']
    print("Conectado a MongoDB exitosamente.")
except Exception as e:
    print(f"!!! ERROR: No se pudo conectar a MongoDB. Error: {e}")
    exit()

def log_event(tipo_evento, **kwargs):
    try:
        evento = {"timestamp": datetime.now(), "tipo_evento": tipo_evento, **kwargs}
        eventos_collection.insert_one(evento)
    except Exception as e:
        print(f"Error al registrar evento en MongoDB: {e}")

# --- CARGA DE MODELOS Y WATCHLISTS ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
known_face_encodings = []
known_face_metadata = []
lock = threading.Lock()

def load_watchlists():
    global known_face_encodings, known_face_metadata
    temp_encodings, temp_metadata = [], []
    print("Cargando watchlists...")
    for category in os.listdir(WATCHLIST_DIR):
        category_path = os.path.join(WATCHLIST_DIR, category)
        if os.path.isdir(category_path):
            for filename in os.listdir(category_path):
                try:
                    image_path = os.path.join(category_path, filename)
                    person_name = os.path.splitext(filename)[0].replace("_", " ").title()
                    person_image = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(person_image)
                    if encodings:
                        temp_encodings.append(encodings[0])
                        temp_metadata.append({'name': person_name, 'category': category})
                except Exception as e:
                    print(f"Error al cargar la imagen de watchlist {filename}: {e}")
    with lock:
        known_face_encodings = temp_encodings
        known_face_metadata = temp_metadata
    print(f"Watchlists cargadas. {len(known_face_encodings)} personas conocidas.")

# --- VARIABLES GLOBALES ---
output_frame = None
latest_captures = deque(maxlen=5)
tracked_persons_info = {}
last_successful_frame_time = datetime.now()

# --- INICIALIZACIÓN DE FLASK Y AUTENTICACIÓN ---
app = Flask(__name__)
def check_auth(username, password): return username == ADMIN_USERNAME and password == ADMIN_PASSWORD
def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return Response('Acceso no autorizado.', 401, {'WWW-Authenticate': 'Basic realm="Login Required"'})
        return f(*args, **kwargs)
    return decorated

# --- HILO DE PROCESAMIENTO DE VIDEO ---
def video_processing_loop():
    global output_frame, latest_captures, tracked_persons_info, last_successful_frame_time
    print("[HILO DE VIDEO]: Iniciando...")
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened(): print("!!! ERROR: No se pudo conectar a la cámara."); return
    print("[HILO DE VIDEO]: Conexión exitosa.")
    face_trackers = {}; next_tracker_id = 0; frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret: print("[HILO DE VIDEO]: Reconectando..."); cap.release(); time.sleep(5); cap = cv2.VideoCapture(RTSP_URL); continue
        with lock: last_successful_frame_time = datetime.now()
        (H, W) = frame.shape[:2]
        cv2.polylines(frame, [ZONA_RESTRINGIDA], isClosed=True, color=(0, 255, 255), thickness=2)
        current_bboxes = {}
        with lock:
            for tid in list(face_trackers.keys()):
                ok, bbox = face_trackers[tid].update(frame)
                if ok:
                    current_bboxes[tid] = bbox
                    if tid in tracked_persons_info:
                        cx, cy = int(bbox[0] + bbox[2]/2), int(bbox[1] + bbox[3]/2)
                        tracked_persons_info[tid]['positions'].append((cx, cy))
                else:
                    del face_trackers[tid]
                    if tid in tracked_persons_info: del tracked_persons_info[tid]
        if frame_count % DETECTION_INTERVAL == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
            for (x, y, w, h) in faces:
                is_tracked = any(abs((x+w/2)-(tx+tw/2)) < tw*0.5 for tx,ty,tw,th in current_bboxes.values())
                if not is_tracked:
                    x1,y1,x2,y2 = max(0,int(x-w*FACE_PADDING)), max(0,int(y-h*FACE_PADDING)), min(W,int(x+w+w*FACE_PADDING)), min(H,int(y+h+h*FACE_PADDING))
                    face_capture = frame[y1:y2, x1:x2]
                    if face_capture.size > 0:
                        rgb_capture = cv2.cvtColor(face_capture, cv2.COLOR_BGR2RGB)
                        face_encodings = face_recognition.face_encodings(rgb_capture)
                        person_name, person_category = "Desconocido", "normal"
                        if face_encodings:
                            with lock: matches = face_recognition.compare_faces(known_face_encodings, face_encodings[0], tolerance=0.55)
                            if True in matches:
                                metadata = known_face_metadata[matches.index(True)]
                                person_name, person_category = metadata['name'], metadata['category']
                        tid = next_tracker_id
                        filename = f"captura_{datetime.now().strftime('%Y%m%d_%H%M%S')}_id{tid}.jpg"
                        file_path = os.path.join(SAVE_DIR, filename)
                        cv2.imwrite(file_path, face_capture)
                        with lock:
                            latest_captures.appendleft(filename)
                            tracked_persons_info[tid] = {'name': person_name, 'category': person_category, 'positions': deque(maxlen=100), 'first_seen': datetime.now(), 'status': 'normal', 'last_alert': datetime.min}
                        log_event('nueva_persona', persona_id=tid, nombre_persona=person_name, categoria_watchlist=person_category, ruta_imagen=filename)
                        if person_category != 'normal' and person_category != 'empleados':
                            log_event('alerta_poi', persona_id=tid, nombre_persona=person_name, categoria_watchlist=person_category, ruta_imagen=filename)
                        tracker = cv2.TrackerKCF_create(); tracker.init(frame, (x, y, w, h)); face_trackers[tid] = tracker; next_tracker_id += 1
        with lock:
            for tid, bbox in current_bboxes.items():
                if tid in tracked_persons_info:
                    info = tracked_persons_info[tid]; info['status'] = 'normal'
                    x, y, w, h = map(int, bbox); punto_base = (int(x + w/2), y + h)
                    if cv2.pointPolygonTest(ZONA_RESTRINGIDA, punto_base, False) >= 0 and info.get('category') != 'empleados':
                        info['status'] = 'zona_restringida'
                        if datetime.now() - info['last_alert'] > timedelta(seconds=60):
                            log_event('alerta_zona_restringida', persona_id=tid, nombre_persona=info['name'], ruta_imagen=list(latest_captures)[0] if latest_captures else None); info['last_alert'] = datetime.now()
                    if (datetime.now() - info['first_seen']).total_seconds() > LOITERING_TIME_SECONDS:
                        if len(info['positions']) > 20 and np.mean(np.std(np.array(info['positions']), axis=0)) < LOITERING_TOLERANCE_PIXELS:
                            if info['status'] == 'normal': info['status'] = 'merodeando'
                            if datetime.now() - info['last_alert'] > timedelta(seconds=60):
                                log_event('alerta_merodeo', persona_id=tid, nombre_persona=info['name'], ruta_imagen=list(latest_captures)[0] if latest_captures else None); info['last_alert'] = datetime.now()
        with lock:
            for tid, bbox in current_bboxes.items():
                x, y, w, h = map(int, bbox)
                info = tracked_persons_info.get(tid, {}); category, status, name = info.get('category', 'normal'), info.get('status', 'normal'), info.get('name', 'Procesando...')
                color, label = (0, 255, 0), f"ID: {tid} - {name}"
                if category == 'ladrones_conocidos': color, label = (0, 0, 255), f"ALERTA POI: {name}"
                elif status == 'zona_restringida': color, label = (0, 140, 255), f"ALERTA ZONA: {name}"
                elif status == 'merodeando': color, label = (0, 255, 255), f"MERODEO: {name}"
                elif category == 'vips': color, label = (255, 255, 0), f"VIP: {name}"
                elif category == 'empleados': color, label = (255, 165, 0), f"Empleado: {name}"
                cv2.rectangle(frame, (x,y), (x+w,y+h), color, 3); cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        frame_count += 1
        with lock: output_frame = frame.copy()

# --- RUTAS DE FLASK ---
@app.route("/")
@requires_auth
def index(): return render_template("index.html")

@app.route("/video_feed")
@requires_auth
def video_feed():
    def generate():
        global output_frame, lock
        while True:
            time.sleep(0.04)
            with lock:
                if output_frame is None: continue
                (flag, encodedImage) = cv2.imencode(".jpg", output_frame)
                if not flag: continue
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/eventos")
@requires_auth
def eventos():
    eventos_data = list(eventos_collection.find().sort("timestamp", -1).limit(100))
    return render_template("eventos.html", eventos=eventos_data)

@app.route("/galeria")
@requires_auth
def galeria():
    with lock:
        image_files = sorted([f for f in os.listdir(SAVE_DIR) if f.endswith('.jpg')], key=lambda x: os.path.getmtime(os.path.join(SAVE_DIR, x)), reverse=True)
    return render_template("gallery.html", images=image_files)

@app.route("/gestion")
@requires_auth
def gestion():
    watchlists_data = {}
    with lock:
        for category in os.listdir(WATCHLIST_DIR):
            category_path = os.path.join(WATCHLIST_DIR, category)
            if os.path.isdir(category_path): watchlists_data[category] = os.listdir(category_path)
    return render_template("gestion.html", watchlists=watchlists_data)

@app.route("/capturas/<path:filename>")
@requires_auth
def servir_captura(filename): return send_from_directory(SAVE_DIR, filename)

@app.route("/watchlists/<category>/<filename>")
@requires_auth
def servir_watchlist_imagen(category, filename): return send_from_directory(os.path.join(WATCHLIST_DIR, category), filename)

@app.route("/api/latest_captures")
@requires_auth
def api_latest_captures():
    with lock: return jsonify(list(latest_captures))

@app.route("/api/classify_face", methods=['POST'])
@requires_auth
def classify_face():
    data = request.json; filename, category = data.get('filename'), data.get('category')
    with lock:
        source_path = os.path.join(SAVE_DIR, filename)
        if not os.path.exists(source_path): return jsonify({"success": False, "error": "El archivo ya no existe."}), 404
        try:
            person_name_in_file = '_'.join(filename.split('_id')[0].split('_')[1:]) or "persona_desconocida"
            new_filename = f"{person_name_in_file}.jpg"
            shutil.move(source_path, os.path.join(WATCHLIST_DIR, category, new_filename))
        except Exception as e: return jsonify({"success": False, "error": str(e)}), 500
    threading.Thread(target=load_watchlists).start()
    return jsonify({"success": True})

@app.route("/api/delete_person", methods=['POST'])
@requires_auth
def delete_person():
    data = request.json; filename, category = data.get('filename'), data.get('category')
    file_path = os.path.join(WATCHLIST_DIR, category, filename)
    with lock:
        if not os.path.exists(file_path): return jsonify({"success": False, "error": "El archivo no existe."}), 404
        try: os.remove(file_path)
        except Exception as e: return jsonify({"success": False, "error": str(e)}), 500
    threading.Thread(target=load_watchlists).start()
    return jsonify({"success": True})

@app.route("/api/modify_person", methods=['POST'])
@requires_auth
def modify_person():
    data = request.json; filename, old_category, new_category = data.get('filename'), data.get('old_category'), data.get('new_category')
    source_path = os.path.join(WATCHLIST_DIR, old_category, filename)
    dest_path = os.path.join(WATCHLIST_DIR, new_category, filename)
    with lock:
        if not os.path.exists(source_path): return jsonify({"success": False, "error": "El archivo no existe."}), 404
        try: shutil.move(source_path, dest_path)
        except Exception as e: return jsonify({"success": False, "error": str(e)}), 500
    threading.Thread(target=load_watchlists).start()
    return jsonify({"success": True})

@app.route("/health")
# @requires_auth
def health_check():
    with lock: time_since_last_frame = (datetime.now() - last_successful_frame_time).total_seconds()
    if time_since_last_frame > 45:
        error_message = f"ALERTA DE SALUD: Sin frames en {time_since_last_frame:.0f}s."
        print(error_message)
        return error_message, 503
    return f"OK: Último frame hace {time_since_last_frame:.0f}s.", 200

# --- PUNTO DE INICIO ---
if __name__ == '__main__':
    load_watchlists()
    video_thread = threading.Thread(target=video_processing_loop, daemon=True)
    video_thread.start()
    print(f"[APP PRINCIPAL]: Iniciando servidor Flask. Accede en http://127.0.0.1:8080")
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)
