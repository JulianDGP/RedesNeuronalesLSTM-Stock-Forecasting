import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import Callback

# Clase de Callback para actualizar la barra de progreso en Streamlit durante el entrenamiento del modelo
class StreamlitProgressCallback(Callback):
    def __init__(self, progress_bar, status_text, epochs):
        super().__init__()
        self.progress_bar = progress_bar # Barra de progreso en la UI de Streamlit
        self.status_text = status_text # Texto de estado en la UI de Streamlit
        self.epochs = epochs # cantidad épocas para el entrenamiento.
    # Método que se ejecuta al final de cada época
    def on_epoch_end(self, epoch, logs=None):
        self.progress_bar.progress((epoch + 1) / self.epochs) # Actualiza la barra de progreso
        self.status_text.text(f'Epoch {epoch + 1}/{self.epochs} - Loss: {logs["loss"]:.4f}') # Actualiza el texto de estado

# Título de la Aplicación en Streamlit
st.title('Predicción del Valor Futuro de las Acciones de Amazon usando Redes Neuronales LSTM.')

# Mostrar la imagen debajo del título
st.image('akinator2.webp', use_column_width=True)

# Añadir información de los creadores con iconos de GitHub y LinkedIn
st.markdown("""
### Hecho por
- **Julian David Gomez**: 
  <a href="https://github.com/JulianDGP" target="_blank"><img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="30"></a> 
  <a href="https://www.linkedin.com/in/julian-david-gomez/" target="_blank"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25"></a>
- **Luis Felipe Rivera**:
  <a href="https://github.com/LFelipe-RiveraH" target="_blank"><img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="30"></a> 
  <a href="https://www.linkedin.com/in/luis-felipe-rivera-hern%C3%A1ndez-8587b4185/" target="_blank"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25"></a> 
""", unsafe_allow_html=True)

# Función para cargar datos de acciones de Amazon desde Yahoo Finance
def cargar_datos():
    # Descarga datos de acciones de Amazon desde el 1 de enero de 2010 hasta la fecha actual
    df = yf.download('AMZN', start='2010-01-01', end=pd.Timestamp.today().strftime('%Y-%m-%d'))
    return df

# Función para crear secuencias de datos para entrenar el modelo LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length]) # Secuencia de entrada
        y.append(data[i + seq_length])  # Valor de salida correspondiente
    return np.array(X), np.array(y)

# Inicializar variables de estado
if 'df' not in st.session_state:
    st.session_state.df = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'train_size' not in st.session_state:
    st.session_state.train_size = None
if 'seq_length' not in st.session_state:
    st.session_state.seq_length = None

# Cargar datos si se presiona el botón
if st.button('Cargar datos actualizados desde Yahoo!'):
    df = cargar_datos() # Cargar set de daos desde Yahoo Finance
    start_date = df.index.min().strftime('%Y-%m-%d')
    end_date = df.index.max().strftime('%Y-%m-%d')
    st.write(f'Datos cargados sin procesar desde {start_date} hasta {end_date}:')
    st.write(df)
    st.session_state.df = df
    st.session_state.model_trained = False
    st.experimental_rerun()

# Mostrar tabla y gráfica si los datos están cargados
if st.session_state.df is not None:
    df = st.session_state.df

    # Mostrar tabla con datos crudos
    st.write(f'Datos cargados sin procesar desde {df.index.min().strftime("%Y-%m-%d")} hasta {df.index.max().strftime("%Y-%m-%d")}:')
    st.write(df)

    # Gráfica de la serie de tiempo
    st.write('Gráfica de la serie de tiempo del valor de las acciones:')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Adj Close'], mode='lines', name='Amazon Stock Price'))
    fig.update_layout(
        title='Serie de Tiempo del Valor de las Acciones de Amazon',
        xaxis_title='Fecha (Días)',
        yaxis_title='Precio de Cierre ajustado (USD)',
        width=1200,
        height=600,
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"
        ),
        xaxis=dict(
            showgrid=True,  # Mostrar líneas de la cuadrícula
            gridcolor='rgba(128, 128, 128, 0.5)',  # Color de la cuadrícula
        ),
        yaxis=dict(
            showgrid=True,  # Mostrar líneas de la cuadrícula
            gridcolor='rgba(128, 128, 128, 0.5)',  # Color de la cuadrícula
        )
    )
    st.plotly_chart(fig, use_container_width=True)
    # Mostrar información de las parametrizaciones
    st.markdown("""
    **Parámetros del Modelo:**

    - **Epochs:** Número de veces que el modelo verá el conjunto de datos completo durante el entrenamiento.
    - **Batch Size:** Número de muestras que se procesan antes de actualizar el modelo. Influye en la memoria y la velocidad de entrenamiento.
    - **Neuronas LSTM (Capa 1):** Número de neuronas en la primera capa LSTM, afecta la capacidad del modelo para aprender patrones complejos.
    - **Neuronas LSTM (Capa 2):** Número de neuronas en la segunda capa LSTM, permite captar características más complejas de los datos.
    """)

    # Lista de opciones con descripciones de tiempo de entrenamiento
    epoch_options = [
        (2, 'epoch ≈ 1 min'),
        (5, 'epochs ≈ 2 min'),
        (25, 'epochs ≈ 2 min'),
        (50, 'epochs ≈ 10 min'),
        (100, 'epochs ≈ 10 min'),
        (150, 'epochs ≈ 15 min'),
        (200, 'epochs ≈ 20 min')
    ]
    # Crear una lista de descripciones para mostrar en el selectbox
    epoch_descriptions = [f'{epoch} {desc}' for epoch, desc in epoch_options]

    # Selección de cantidad de epochs
    selected_description = st.selectbox('Selecciona la cantidad de epochs para entrenar el modelo:', epoch_descriptions, index=2)

    # Extraer el valor numérico de la opción seleccionada
    selected_epochs = int(selected_description.split(' ')[0])

    # Selección de tamaño de batch
    selected_batch_size = st.selectbox('Selecciona el tamaño del batch:', [4, 8, 16, 32, 64, 128, 256], index=4)

    # Selección de neuronas en la primera capa LSTM
    lstm_neurons_1 = st.selectbox('Selecciona el número de neuronas en la primera capa LSTM:', [10, 25, 50, 100], index=2)

    # Selección de neuronas en la segunda capa LSTM
    lstm_neurons_2 = st.selectbox('Selecciona el número de neuronas en la segunda capa LSTM:', [10, 25, 50, 100], index=2)

    # Botón para entrenar el modelo
    if st.button('Entrenar modelo'):
        progress_bar = st.progress(0) # Inicializar barra de progreso
        status_text = st.empty() # Inicializar texto de estado

        # Preprocesar los datos
        data = df[['Adj Close']]
        data = data.dropna() # Usar solo la columna de precios de cierre ajustados
        scaler = MinMaxScaler(feature_range=(0, 1)) # Escalador de MinMax
        scaled_data = scaler.fit_transform(data) # Escalar los datos

        # Dividir en entrenamiento y prueba
        train_size = int(len(scaled_data) * 0.9) # 90% para entrenamiento
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size:]

        seq_length = 60 # Longitud de la secuencia para LSTM
        X_train, y_train = create_sequences(train_data, seq_length) # Crear secuencias para entrenamiento
        X_test, y_test = create_sequences(test_data, seq_length) # Crear secuencias para prueba

        # Ajustar la forma de los datos para LSTM
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        # Construir el modelo LSTM
        model = Sequential()
        model.add(LSTM(lstm_neurons_1, return_sequences=True, input_shape=(seq_length, 1)))  # Primera capa LSTM
        model.add(LSTM(lstm_neurons_2, return_sequences=False)) # Segunda capa LSTM
        model.add(Dense(25)) # Capa densa con 25 neuronas
        model.add(Dense(1)) # Capa de salida con una neurona

        # Compilar y entrenar el modelo
        model.compile(optimizer='adam', loss='mean_squared_error')  # Compilación del modelo
        model.fit(X_train, y_train,
                  batch_size=selected_batch_size,
                  epochs=selected_epochs,
                  callbacks=[StreamlitProgressCallback(progress_bar, status_text, selected_epochs)]) # Entrenamiento del modelo

        # Hacer predicciones
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions) # Desescalar las predicciones

        # Crear índice de validación
        valid = df.iloc[train_size + seq_length:].copy()
        valid['Predictions'] = predictions

        # Guardar el estado del modelo y las predicciones
        st.session_state.model_trained = True
        st.session_state.predictions = valid
        st.session_state.scaler = scaler
        st.session_state.model = model
        st.session_state.train_size = train_size
        st.session_state.seq_length = seq_length
        st.experimental_rerun() # Recargar la página para reflejar los cambios

    # Mostrar predicciones si el modelo ha sido entrenado
    if st.session_state.model_trained:
        valid = st.session_state.predictions

        # Mostrar tabla con valores reales y predichos
        st.write('Tabla de valores reales y predichos:')
        resultados = valid[['Adj Close', 'Predictions']]
        resultados.columns = ['Valor Real', 'Valor Predicho']
        st.write(resultados)

        # Gráfica de predicción vs. realidad
        st.write('Gráfica de predicción vs. realidad:')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index[:st.session_state.train_size + st.session_state.seq_length], y=df['Adj Close'][:st.session_state.train_size + st.session_state.seq_length], mode='lines', name='Entrenamiento', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=valid.index, y=valid['Adj Close'], mode='lines', name='Real', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=valid.index, y=valid['Predictions'], mode='lines', name='Predicción', line=dict(color='green')))
        fig.update_layout(
            title='Predicción del Precio de las Acciones de Amazon',
            xaxis_title='Fecha (Días)',
            yaxis_title='Precio de Cierre ajustado (USD)',
            width=1200,
            height=600,
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f"
            ),
            xaxis=dict(
            showgrid=True,  # Mostrar líneas de la cuadrícula
            gridcolor='rgba(128, 128, 128, 0.5)',  # Color de la cuadrícula
        ),
        yaxis=dict(
            showgrid=True,  # Mostrar líneas de la cuadrícula
            gridcolor='rgba(128, 128, 128, 0.5)',  # Color de la cuadrícula
        )
        )
        st.plotly_chart(fig, use_container_width=True)

        # Sección para predicción de días futuros
        st.write('### Hagamos la predicción de los siguientes días')
        days_to_predict = st.slider('Selecciona el número de días para predecir:', 1, 30, 1)

        if st.button('Predecir ahora'):
            # Usar los datos completos para la predicción futura
            last_sequence = st.session_state.scaler.transform(df[['Adj Close']].values[-st.session_state.seq_length:])
            future_predictions = []

            for _ in range(days_to_predict):
                prediction = st.session_state.model.predict(last_sequence.reshape(1, st.session_state.seq_length, 1))
                future_predictions.append(prediction[0, 0])
                last_sequence = np.append(last_sequence[1:], prediction, axis=0)

            future_predictions = st.session_state.scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

            # Crear las fechas futuras
            last_date = df.index[-1]
            future_dates = pd.date_range(last_date, periods=days_to_predict + 1, freq='D')[1:]

            # Crear un DataFrame con los resultados
            future_df = pd.DataFrame({'Fecha': future_dates, 'Valor Predicho': future_predictions.flatten()})
            st.write('Predicciones para los próximos días:')
            st.write(future_df)

            # Recomendaciones de compra/venta
            recomendaciones = ['Comprar' if future_predictions[i] > df['Adj Close'].iloc[-1] else 'Vender' for i in range(len(future_predictions))]
            future_df['Recomendación'] = recomendaciones
            st.write('Predicciones y recomendaciones:')
            st.write(future_df)
else:
    st.write('Por favor, carga los datos para ver la gráfica.')