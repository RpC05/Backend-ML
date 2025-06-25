from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import uuid
import os
import io
import threading
import requests
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from datetime import datetime, timedelta, timezone
from supabase import create_client, Client

# --- 1. CONFIGURACIÓN DE LA APP Y SUPABASE ---
load_dotenv()
app = Flask(__name__)
CORS(app)
 
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY") 
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Las variables de entorno SUPABASE_URL y SUPABASE_KEY son requeridas.")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
print("Cliente de Supabase inicializado.")

# Variables globales y constantes
BUCKET_NAME = "modelos"
X_columns = None
models = {}
platos = ['Aji de Gallina', 'Arroz Chaufa', 'Arroz con Chancho', 'Caldo de Gallina', 'Causa de Ceviche',
          'Causa de Pollo', 'Ceviche', 'Chicharron de Pescado', 'Chicharron de Pollo', 'Churrasco',
          'Cuy Guisado', 'Gallo Guisado', 'Lomo Saltado', 'Pescado Frito', 'Pollo Guisado', 'Rocoto Relleno',
          'Seco de Cordero', 'Sudado de Pescado', 'Tacu Tacu', 'Tallarines Verdes con Chuleta']
MAE_THRESHOLD = 5
MONITORING_WINDOW_DAYS = 7
MIN_SAMPLES_FOR_RETRAIN = 5

def load_models_from_storage():
    global X_columns, models
    print("Iniciando carga de modelos desde Supabase Storage...")
    try:
        # Cargar columnas de entrenamiento
        file_bytes = supabase.storage.from_(BUCKET_NAME).download("columnas_entrenamiento.pkl")
        X_columns = joblib.load(io.BytesIO(file_bytes))
        print("Columnas de entrenamiento cargadas.")

        # Cargar modelo para cada plato
        for plato in platos:
            model_file_name = f'modelo_{plato.replace(" ", "_")}.pkl'
            try:
                file_bytes = supabase.storage.from_(BUCKET_NAME).download(model_file_name)
                models[plato] = joblib.load(io.BytesIO(file_bytes))
                print(f"Modelo cargado para {plato}")
            except Exception as e:
                # Maneja el caso en que un modelo no exista en el bucket
                print(f"Advertencia: No se encontró modelo para {plato} en el bucket. Error: {e}")
                models[plato] = None
    except Exception as e:
        print(f"ERROR CRÍTICO: No se pudieron cargar los modelos iniciales. {e}")

# Cargar los modelos una sola vez al iniciar la aplicación
load_models_from_storage()

# Endpoint de salud para verificar que la app está viva
@app.route('/', methods=['GET'])
def health_check():
    return "Backend para Predicciones Doña Bere está activo."

# --- 3. ENDPOINT DE PREDICCIÓN CON REGISTRO EN SUPABASE ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        prediction_group_id = str(uuid.uuid4())
        input_data_df = pd.DataFrame([data])
        processed_input = pd.get_dummies(input_data_df, columns=['nombre_dia', 'clima'])
        missing_cols = set(X_columns) - set(processed_input.columns)
        for col in missing_cols:
            processed_input[col] = 0
        processed_input = processed_input[X_columns]
        predictions_response = {}
        supabase_records = []
        total_platos = 0
        for plato, model in models.items():
            if model is None:
                predicted_quantity = 0
            else:
                predicted_quantity = max(0, round(model.predict(processed_input)[0]))
            predictions_response[plato] = int(predicted_quantity)
            total_platos += predicted_quantity
            supabase_records.append({
                'prediction_group_id': prediction_group_id,
                'dish_name': plato,
                'input_data': data,
                'predicted_quantity': int(predicted_quantity)
            })
        supabase.table('dish_predictions').insert(supabase_records).execute()
        predictions_response['total_platos'] = int(total_platos)
        predictions_response['prediction_group_id'] = prediction_group_id
        return jsonify(predictions_response)
    except Exception as e:
        print(f"Error en /predict: {e}")
        return jsonify({'error': str(e)}), 500

# --- 4. ENDPOINT DE FEEDBACK ---
@app.route('/feedback', methods=['POST'])
def feedback():
    try:
        feedback_data = request.get_json()
        prediction_group_id = feedback_data.get('prediction_group_id')
        observed_sales = feedback_data.get('observed_sales')
        if not prediction_group_id or not observed_sales:
            return jsonify({'error': 'Faltan prediction_group_id u observed_sales'}), 400
        response = supabase.table('dish_predictions').select('id, predicted_quantity, dish_name').eq('prediction_group_id', prediction_group_id).execute()
        if not response.data:
            return jsonify({'error': 'prediction_group_id no encontrado'}), 404
        for record in response.data:
            dish_name = record['dish_name']
            observed_quantity = observed_sales.get(dish_name, 0)
            predicted_quantity = record['predicted_quantity']
            absolute_error = abs(predicted_quantity - observed_quantity)
            supabase.table('dish_predictions').update({
                'observed_quantity': observed_quantity,
                'absolute_error': absolute_error
            }).eq('id', record['id']).execute()

        base_url = request.url_root 
        # Llama al endpoint de reentrenamiento en un hilo separado para no esperar
        retrain_thread = threading.Thread(target=trigger_retrain_request, args=(base_url,))
        retrain_thread.start()

        print(f"Feedback procesado. Disparando reentrenamiento en segundo plano.")
        return jsonify({'message': 'Feedback procesado con éxito. El reentrenamiento ha comenzado.'}), 200
    
    except Exception as e:
        print(f"Error en /feedback: {e}")
        return jsonify({'error': str(e)}), 500

def trigger_retrain_request(base_url):
    try:
        requests.post(f"{base_url}trigger-retrain", timeout=3)
        print("Petición a /trigger-retrain enviada.")
    except requests.exceptions.ReadTimeout:
        print("La petición a /trigger-retrain ha expirado (comportamiento esperado).")
    except Exception as e:
        print(f"Error al disparar el reentrenamiento: {e}")



# --- 5. LÓGICA Y ENDPOINT DEL PIPELINE DE REENTRENAMIENTO ---

def send_alert(message):
    # En Vercel, las alertas se verán en los logs de la función
    print("🚨 ALERTA 🚨")
    print(message)

def monitor_performance(df):
    """Monitorea el error (MAE) por plato en los datos recibidos."""
    print("\n--- Iniciando Monitoreo de Rendimiento ---")
    df['created_at'] = pd.to_datetime(df['created_at'])
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=MONITORING_WINDOW_DAYS)
    recent_data = df[df['created_at'] >= cutoff_date]

    if recent_data.empty:
        print("No hay datos recientes para monitorear.")
        return

    try:
        mae_by_dish = recent_data.groupby('dish_name').apply(
            lambda x: mean_absolute_error(x['observed_quantity'], x['predicted_quantity'])
        )
        print("Error Absoluto Medio (MAE) por plato en la última semana:")
        print(mae_by_dish)
        for dish, mae in mae_by_dish.items():
            if mae > MAE_THRESHOLD:
                send_alert(f"El MAE para '{dish}' es {mae:.2f}, superando el umbral de {MAE_THRESHOLD}.")
    except Exception as e:
        print(f"Error durante el monitoreo: {e}")


def retrain_and_upload_models(df):
    """Reentrena, sube y recarga en memoria los modelos mejorados."""
    global X_columns, models # Indicar que modificaremos las variables globales
    print("\n--- Iniciando Pipeline de Reentrenamiento y Subida ---")

    platos_a_reentrenar = df['dish_name'].unique()

    for plato in platos_a_reentrenar:
        print(f"\nProcesando plato: {plato}")
        plato_data = df[df['dish_name'] == plato].copy()

        # Agrupar por día para tener una muestra por día y evitar sesgos
        daily_samples = plato_data.groupby(plato_data['created_at'].dt.date).first().reset_index()

        if len(daily_samples) < MIN_SAMPLES_FOR_RETRAIN:
            print(f"Datos insuficientes ({len(daily_samples)}) para reentrenar '{plato}'. Se necesitan {MIN_SAMPLES_FOR_RETRAIN}.")
            continue
        
        # --- Lógica de preparación de datos (AHORA DENTRO DE LA FUNCIÓN) ---
        y = daily_samples['observed_quantity']
        X = pd.get_dummies(daily_samples, columns=['nombre_dia', 'clima'])
        missing_cols = set(X_columns) - set(X.columns)
        for col in missing_cols:
            X[col] = 0
        X = X[X_columns]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Cargar modelo antiguo desde la memoria para comparar
        old_model = models.get(plato)
        if old_model:
            old_mae = mean_absolute_error(y_test, old_model.predict(X_test))
            print(f"MAE del modelo antiguo para '{plato}': {old_mae:.2f}")
        else:
            old_mae = float('inf')
            print(f"No se encontró modelo antiguo en memoria para '{plato}'.")

        # Reentrenar el nuevo modelo
        # NOTA: Los hiperparámetros de XGBRegressor estaban como (...)
        # Los he puesto con valores razonables. Ajústalos si es necesario.
        new_model = XGBRegressor(colsample_bytree=0.8, learning_rate=0.05, max_depth=4, n_estimators=100, subsample=0.8)
        new_model.fit(X_train, y_train)
        new_mae = mean_absolute_error(y_test, new_model.predict(X_test))
        print(f"MAE del modelo nuevo para '{plato}': {new_mae:.2f}")

        # Subir el modelo nuevo solo si es mejor
        if new_mae < old_mae:
            print(f"El nuevo modelo para '{plato}' es mejor. Subiendo y actualizando...")
            model_file_name = f'modelo_{plato.replace(" ", "_")}.pkl'
            
            # Guardar el modelo en un buffer de memoria para subirlo
            model_buffer = io.BytesIO()
            joblib.dump(new_model, model_buffer)
            model_buffer.seek(0)

            try:
                supabase.storage.from_(BUCKET_NAME).upload(
                    file=model_buffer,
                    path=model_file_name,
                    file_options={'cache-control': '3600', 'upsert': 'true'}
                )
                print(f"Modelo '{model_file_name}' actualizado en Supabase Storage.")
                
                # --- PASO CRÍTICO: Actualizar el modelo en la memoria de la app ---
                models[plato] = new_model
                print(f"Modelo para '{plato}' actualizado en memoria.")

            except Exception as e:
                print(f"Error al subir el modelo a Supabase Storage: {e}")
        else:
            print(f"El modelo antiguo para '{plato}' es mejor o igual. No se actualiza.")


@app.route('/trigger-retrain', methods=['POST'])
def trigger_retrain(): 
    print("\n--- INICIANDO PIPELINE DE REENTRENAMIENTO (disparado por webhook) ---")
    try:
        # 1. Obtener todos los datos con feedback de Supabase
        response = supabase.table('dish_predictions').select('*').not_.is_('observed_quantity', 'NULL').execute()
        if not response.data:
            print("No hay datos con feedback para reentrenar.")
            return jsonify({'message': 'No hay datos nuevos'}), 200
        
        df = pd.DataFrame(response.data)
        input_df = pd.json_normalize(df['input_data'])
        all_data = pd.concat([input_df.reset_index(drop=True), df[['dish_name', 'predicted_quantity', 'observed_quantity', 'created_at']].reset_index(drop=True)], axis=1)
        print(f"Se descargaron {len(all_data)} registros con feedback.")
        
        # 2. Ejecutar el monitoreo y el reentrenamiento
        monitor_performance(all_data)
        retrain_and_upload_models(all_data)
        
        print("\n--- PIPELINE DE REENTRENAMIENTO FINALIZADO ---")
        return jsonify({'message': 'Reentrenamiento completado'}), 200
    except Exception as e:
        print(f"ERROR DURANTE EL REENTRENAMIENTO: {e}")
        return jsonify({'error': str(e)}), 500
