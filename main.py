from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import librosa
import io

# Cargamos modelo entrenado
modelo = tf.keras.models.load_model("modelo_yamnet.h5")

# Cargamos YAMNet
yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

# Lista de clases 
CLASES = ["Ardea Alba", "Azara Sspinetail", "Black Facedantbird", "Bubulcus ibis", "Butorides striata", "Grey Breastedwoodwren", 
          "Housewren", "Southernchestnut Tailedantbird", "White Browedantbird", ]  

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Validamos formato de archivo
    if not file.filename.lower().endswith((".wav", ".mp3", ".ogg")):
        return {"error": "Formato de archivo no soportado. Sube WAV, MP3 u OGG."}
    
    # Leemos el archivo
    audio_bytes = await file.read()

    # Carga audio con librosa
    try:
        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
    except Exception as e:
        return {"error": f"No se pudo procesar el audio: {str(e)}"}

    # Convertimos a tensor
    audio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)

    # Obtenemos embedding con YAMNet
    _, embeddings, _ = yamnet(audio_tensor)
    embedding = embeddings.numpy().mean(axis=0)

    # Preparamos el input para el modelo
    input_array = np.expand_dims(embedding, axis=0)

    # Predicción
    preds = modelo.predict(input_array)
    idx_pred = np.argmax(preds[0])
    confidence = float(preds[0][idx_pred])

    # Umbral de confianza
    UMBRAL = 0.5 
    if confidence < UMBRAL:
        clase_predicha = "Desconocido"
    else:
        clase_predicha = CLASES[idx_pred]

    return {
        "predicted_class": clase_predicha,
        "confidence": confidence
    }

