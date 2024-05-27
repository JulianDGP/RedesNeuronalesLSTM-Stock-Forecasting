import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Título de la Aplicación
st.title('Predicción del Valor Futuro de las Acciones de Amazon usando Redes Neuronales LSTM!')

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

# Función para cargar datos
def cargar_datos():
    df = yf.download('AMZN', start='2010-01-01', end=pd.Timestamp.today().strftime('%Y-%m-%d'))
    return df

# Función para crear secuencias
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
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
if st.button('Cargar datos mas recientes'):
    df = cargar_datos()
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
        xaxis_rangeslider_visible=True,
        width=1200,
        height=600,
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#7f7f7f"
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    # Botón para entrenar el modelo
    if st.button('Entrenar modelo'):
        # Preprocesar los datos
        data = df[['Adj Close']]
        data = data.dropna()
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # Dividir en entrenamiento y prueba
        train_size = int(len(scaled_data) * 0.9)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size:]

        seq_length = 60
        X_train, y_train = create_sequences(train_data, seq_length)
        X_test, y_test = create_sequences(test_data, seq_length)

        # Ajustar la forma de los datos para LSTM
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        # Construir el modelo LSTM
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))

        # Compilar y entrenar el modelo
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, batch_size=1, epochs=1)

        # Hacer predicciones
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)

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
        st.experimental_rerun()

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

else:
    st.write('Por favor, carga los datos para ver la gráfica.')