# Redes Neuronales LSTM para Predicción de series temporales

## 👨‍💻 Creadores
Este proyecto fue creado por:
* [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/JulianDGP) **Julián**
* [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/LFelipe-RiveraH) **Felipe**

## 📝 Descripción

Este proyecto utiliza redes neuronales LSTM para tratar de hacer predicciones en el mercado de acciones. Se centra en analizar series temporales históricas de precios de acciones y generar predicciones futuras.

## ⚠️ Requisitos

- Python 3.8 o superior
- pip (Administrador de paquetes de Python)

## 🛠️Instalación

1. Clonar el repositorio:
```
git clone https://github.com/JulianDGP/RedesNeuronalesLSTM-Stock-Forecasting
```
2. Navegar al directorio local del proyecto
```
cd RedesNeuronalesLSTM-Stock-Forecasting
```
3. Crear un entorno virtual de Python (recomendado, pero opcional)
```
python -m venv venv
```
4. Activa el entorno virtual:
   - En Windows:
    ```
    .\venv\Scripts\activate
    ```
    - En macOS/Linux:
    ```
    source venv/bin/activate
    ```
5. Instalar las dependencias del proyecto:
```
pip install -r requirements.txt
```
## 🚀 Ejecución
Para ejecutar el proyecto y visualizar la aplicación Streamlit
1. Asegúrate de que el entorno virtual esté activado.
2. Ejecuta el siguiente comando:
```
streamlit run app.py
```
Esto iniciará la aplicación y estará accesible en el navegador web a través de la dirección http://localhost:8501, para cambiar el puerto de ejecucion de Streamlit:
```
streamlit run app.py --server.port 8080
```