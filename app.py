import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, render_template

app = Flask(__name__)

try:
    paquete_cargado = joblib.load('AirPolutionModel.joblib')
except FileNotFoundError as e:
    raise FileNotFoundError("Archivo 'AirPolutionModel.joblib' no encontrado.") from e

modelo_cargado = paquete_cargado['modelo']
imputador_cargado = paquete_cargado['imputador_num']
escalador_cargado = paquete_cargado['escalador']
cols_num = paquete_cargado['columnas_numericas']
cols_cat = paquete_cargado['columnas_categoricas']
cols_finales = paquete_cargado['columnas_finales_entrenamiento']


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        so2_val = request.form.get('so2', '')
        so2 = np.nan if so2_val.strip() == '' else float(so2_val)

        no2 = float(request.form['no2'])
        rspm = float(request.form['rspm'])
        spm = float(request.form['spm'])

        type_val = request.form['type']
        state_val = request.form['state']

        datos_nuevos_raw = pd.DataFrame([{
            'so2': so2,
            'no2': no2,
            'rspm': rspm,
            'spm': spm,
            'type': type_val,
            'state': state_val
        }])

        datos_nuevos_imputados = datos_nuevos_raw.copy()
        datos_nuevos_imputados[cols_num] = imputador_cargado.transform(datos_nuevos_imputados[cols_num])
        datos_nuevos_imputados[cols_num] = escalador_cargado.transform(datos_nuevos_imputados[cols_num])

        datos_nuevos_codificados = pd.get_dummies(datos_nuevos_imputados, columns=cols_cat, dtype=int)

        datos_finales_listos = datos_nuevos_codificados.reindex(
            columns=cols_finales,
            fill_value=0
        )

        prediccion = modelo_cargado.predict(datos_finales_listos)
        resultado_final = prediccion[0]

        prediction_str = f'{resultado_final:.2f}'
        app.config['LAST_PREDICTION'] = prediction_str

        return render_template('index.html', prediction_result=prediction_str)

    except ValueError:
        return render_template('index.html', prediction_result="Error: Ingresa valores numéricos válidos.")
    except Exception as e:
        return render_template('index.html', prediction_result=f"Error: {e}")
    
if __name__ == '__main__':
    app.run(debug=True)