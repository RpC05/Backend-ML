from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import uuid
import os
import io
import threading
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
X_columns_template = None # DataFrame plantilla para asegurar la estructura
models = {}
platos = ['Aji de Gallina', 'Arroz Chaufa', 'Arroz con Chancho', 'Caldo de Gallina',
          'Causa de Pollo', 'Ceviche', 'Chicharron de Pescado', 'Chicharron de Pollo', 'Churrasco',
          'Cuy Guisado', 'Gallo Guisado', 'Lomo Saltado', 'Pescado Frito', 'Pollo Guisado', 'Rocoto Relleno',
          'Seco de Cordero', 'Sudado de Pescado', 'Tacu Tacu', 'Tallarines Verdes con Chuleta']
MAE_THRESHOLD = 5
MONITORING_WINDOW_DAYS = 7
MIN_SAMPLES_FOR_RETRAIN = 5

def load_initial_config():
    global X_columns_template
    print("Iniciando carga de configuración inicial (solo columnas)...")
    try:
        file_bytes = supabase.storage.from_(BUCKET_NAME).download("columnas_entrenamiento.pkl")
        loaded_object = joblib.load(io.BytesIO(file_bytes))
        
        if isinstance(loaded_object, pd.DataFrame):
            column_names = loaded_object.columns
        elif isinstance(loaded_object, pd.Index):
            column_names = loaded_object
        else:
            column_names = list(loaded_object)
            
        X_columns_template = pd.DataFrame(columns=column_names)
        print(f"Columnas de entrenamiento cargadas y estandarizadas con {len(X_columns_template.columns)} columnas.") 
    except Exception as e:
        print(f"ERROR CRÍTICO: No se pudieron cargar las columnas de entrenamiento. La app no puede funcionar. {e}")
        X_columns_template = None

def get_model(plato_name):
    global models
    if plato_name in models:
        return models[plato_name]

    print(f"Modelo para '{plato_name}' no en caché. Descargando...")
    model_file_name = f'modelo_{plato_name.replace(" ", "_")}.pkl'
    try:
        file_bytes = supabase.storage.from_(BUCKET_NAME).download(model_file_name)
        model = joblib.load(io.BytesIO(file_bytes))
        models[plato_name] = model
        print(f"Modelo para '{plato_name}' cargado y cacheado.")
        return model
    except Exception as e:
        print(f"Advertencia: No se pudo cargar el modelo para {plato_name}. Error: {e}")
        models[plato_name] = None
        return None

load_initial_config()

def _prepare_data_for_model(df):
    """Función unificada para preparar datos para predicción o entrenamiento."""
    processed_df = pd.get_dummies(df, columns=['nombre_dia', 'clima'])
    # .reindex() es la forma correcta y rápida de alinear columnas
    final_df = processed_df.reindex(columns=X_columns_template.columns, fill_value=0)
    return final_df

@app.route('/', methods=['GET'])
def health_check():
    return "Backend para Predicciones Doña Bere está activo."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if X_columns_template is None:
            return jsonify({'error': 'La configuración del modelo no está lista. Intente de nuevo más tarde.'}), 503

        prediction_group_id = str(uuid.uuid4())
        input_data_df = pd.DataFrame([data])
        final_input = _prepare_data_for_model(input_data_df)
        
        predictions_response = {}
        supabase_records = []
        total_platos = 0
        
        for plato in platos:
            model = get_model(plato)
            predicted_quantity = 0
            if model:
                try:
                    predicted_quantity = max(0, round(model.predict(final_input)[0]))
                except Exception as e:
                    print(f"Error prediciendo para {plato}: {e}")
            
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

def run_full_retraining_pipeline():
    with app.app_context():
        print("\n--- INICIANDO PIPELINE DE REENTRENAMIENTO (Ejecución directa en hilo) ---")
        try:
            response = supabase.table('dish_predictions').select('*').not_.is_('observed_quantity', 'NULL').execute()
            if not response.data:
                print("PIPELINE: No hay datos con feedback para reentrenar. Finalizando.")
                return

            df = pd.DataFrame(response.data)
            input_df = pd.json_normalize(df['input_data'])
            
            if 'created_at' in input_df.columns:
                input_df = input_df.drop('created_at', axis=1)
                
            all_data = pd.concat([input_df.reset_index(drop=True), df[['dish_name', 'predicted_quantity', 'observed_quantity', 'created_at']].reset_index(drop=True)], axis=1)
            print(f"PIPELINE: Se descargaron {len(all_data)} registros con feedback.")
            
            platos_a_reentrenar = monitor_performance(all_data)
            if not platos_a_reentrenar:
                print("PIPELINE: Ningún modelo supera el umbral de MAE. No se necesita reentrenamiento.")
            else:
                retrain_and_upload_models(all_data, platos_a_reentrenar)
            
            print("\n--- PIPELINE DE REENTRENAMIENTO FINALIZADO CON ÉXITO ---")
        except Exception as e:
            print(f"ERROR CATASTRÓFICO DURANTE EL PIPELINE DE REENTRENAMIENTO: {e}")

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

        retrain_thread = threading.Thread(target=run_full_retraining_pipeline)
        retrain_thread.start()

        print("Feedback procesado. Disparando reentrenamiento directamente en segundo plano.")
        return jsonify({'message': 'Feedback procesado con éxito. El reentrenamiento se ha iniciado en segundo plano.'}), 200
    
    except Exception as e:
        print(f"Error en /feedback: {e}")
        return jsonify({'error': str(e)}), 500

def send_alert(message):
    print(f"ALERTA: {message}")

def monitor_performance(df):
    print("\n--- Iniciando Monitoreo de Rendimiento ---")
    df['created_at'] = pd.to_datetime(df['created_at'])
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=MONITORING_WINDOW_DAYS)
    recent_data = df[df['created_at'] >= cutoff_date]

    if recent_data.empty:
        print("MONITOR: No hay datos recientes para monitorear.")
        return []

    platos_malos = []
    try:
        mae_by_dish = recent_data.groupby('dish_name').apply(
            lambda x: mean_absolute_error(x['observed_quantity'], x['predicted_quantity'])
        ).sort_values(ascending=False)
        print("MONITOR: Error Absoluto Medio (MAE) por plato en la última semana:")
        print(mae_by_dish)
        for dish, mae in mae_by_dish.items():
            if mae > MAE_THRESHOLD:
                send_alert(f"El MAE para '{dish}' es {mae:.2f}, superando el umbral de {MAE_THRESHOLD}.")
                platos_malos.append(dish)
    except Exception as e:
        print(f"MONITOR: Error durante el monitoreo de rendimiento: {e}")
    return platos_malos

def retrain_and_upload_models(df, platos_a_reentrenar):
    global X_columns_template, models
    print("\n--- Iniciando Pipeline de Reentrenamiento y Subida ---") 

    for plato in platos_a_reentrenar:
        print(f"\nRE-TRAIN: Procesando plato: {plato}")
        plato_data = df[df['dish_name'] == plato].copy()
        daily_samples = plato_data.groupby(plato_data['created_at'].dt.date).first().reset_index(drop=True)

        if len(daily_samples) < MIN_SAMPLES_FOR_RETRAIN:
            print(f"RE-TRAIN: Datos insuficientes ({len(daily_samples)}) para reentrenar '{plato}'. Se necesitan {MIN_SAMPLES_FOR_RETRAIN}.")
            continue
        
        y = daily_samples['observed_quantity'].reset_index(drop=True)
        
        # --- CORRECCIÓN FINAL Y DEFINITIVA DE LA LÓGICA ---
        # Preparamos los datos de entrenamiento usando la misma función robusta que para predecir.
        # Esto es rápido, correcto y elimina el error del bucle de reinicio.
        X = _prepare_data_for_model(daily_samples)
        # --- FIN DE LA CORRECCIÓN ---

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        old_model = get_model(plato)
        old_mae = float('inf')
        if old_model:
            try:
                old_mae = mean_absolute_error(y_test, old_model.predict(X_test))
                print(f"RE-TRAIN: MAE del modelo antiguo para '{plato}': {old_mae:.2f}")
            except Exception as e:
                print(f"RE-TRAIN: No se pudo evaluar modelo antiguo: {e}")
        else:
            print(f"RE-TRAIN: No se encontró modelo antiguo en memoria.")

        new_model = XGBRegressor(colsample_bytree=0.8, learning_rate=0.05, max_depth=4, n_estimators=100, subsample=0.8)
        new_model.fit(X_train, y_train)
        new_mae = mean_absolute_error(y_test, new_model.predict(X_test))
        print(f"RE-TRAIN: MAE del modelo nuevo para '{plato}': {new_mae:.2f}")

        if new_mae < old_mae:
            print(f"RE-TRAIN: El nuevo modelo para '{plato}' es mejor. Subiendo y actualizando...")
            model_file_name = f'modelo_{plato.replace(" ", "_")}.pkl'
            
            model_buffer = io.BytesIO()
            joblib.dump(new_model, model_buffer)
            model_buffer.seek(0)

            try:
                supabase.storage.from_(BUCKET_NAME).upload(
                    file=model_buffer,
                    path=model_file_name,
                    file_options={'cache-control': '3600', 'upsert': 'true'}
                )
                print(f"RE-TRAIN: Modelo '{model_file_name}' actualizado en Supabase Storage.")
                models[plato] = new_model
                print(f"RE-TRAIN: Modelo para '{plato}' actualizado en memoria.")
            except Exception as e:
                print(f"RE-TRAIN: Error al subir modelo a Supabase Storage: {e}")
        else:
            print(f"RE-TRAIN: El modelo antiguo es mejor o igual. No se actualiza.")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))