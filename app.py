from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import uuid
from supabase import create_client, Client
import os
import io # Necesario para leer los modelos desde bytes

# --- 1. CONFIGURACIÓN DE LA APP Y SUPABASE ---
app = Flask(__name__)
CORS(app)
 
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
print("Cliente de Supabase inicializado.")

# --- 2. CARGA DE MODELOS Y COLUMNAS DESDE SUPABASE STORAGE ---
BUCKET_NAME = "modelos"

platos = ['Aji de Gallina', 'Arroz Chaufa', 'Arroz con Chancho', 'Caldo de Gallina', 'Causa de Ceviche',
          'Causa de Pollo', 'Ceviche', 'Chicharron de Pescado', 'Chicharron de Pollo', 'Churrasco',
          'Cuy Guisado', 'Gallo Guisado', 'Lomo Saltado', 'Pescado Frito', 'Pollo Guisado', 'Rocoto Relleno',
          'Seco de Cordero', 'Sudado de Pescado', 'Tacu Tacu', 'Tallarines Verdes con Chuleta']

models = {}

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
        return jsonify({'message': 'Feedback procesado con éxito'}), 200
    except Exception as e:
        print(f"Error en /feedback: {e}")
        return jsonify({'error': str(e)}), 500
        
# Endpoint de salud para verificar que la app está viva
@app.route('/', methods=['GET'])
def health_check():
    return "Backend para Predicciones Doña Bere está activo."
