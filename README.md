# Redes Neuronales LSTM para Predicción de series temporales

## 👨‍💻 Creadores
Este proyecto fue creado por:
Este proyecto fue creado por:
* [<img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="18" height="18"> Julián Gomez](https://github.com/JulianDGP)
* [<img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="18" height="18"> Felipe Rivera](https://github.com/LFelipe-RiveraH)

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
Esto iniciará la aplicación y estará accesible en el navegador web a través de la dirección http://localhost:8501, para ejecutar la aplicacion cambiando el puerto de ejecución de Streamlit debe usarse:
```
streamlit run app.py --server.port 8080
```