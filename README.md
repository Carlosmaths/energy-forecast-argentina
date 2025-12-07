# ⚡ Energy Demand Forecast (SADI Argentina)

![Python](https://img.shields.io/badge/Python-3.9-blue)
![TensorFlow](https://img.shields.io/badge/AI-TensorFlow-orange)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![Docker](https://img.shields.io/badge/Deploy-Docker-2496ED)

This project implements a **Recurrent Neural Network (LSTM)** to predict the electrical energy demand of the Argentine Interconnection System (SADI) based on historical hourly data.

🚀 **[VIEW LIVE DEMO HERE](https://energy-forecast-argentina.onrender.com)**

---

## 🧠 About the Project

The goal of this project is to anticipate energy load to assist in operational planning. The model analyzes past 24-hour time windows to predict consumption for the immediate next hour.

### Key Features:
* **Data Engineering:** Automated cleaning of raw government reports (CAMMESA) and data normalization (MinMaxScaling).
* **Deep Learning Model:** LSTM (Long Short-Term Memory) architecture designed to capture complex temporal patterns (daily and seasonal cycles).
* **Interactive App:** A web interface built with Streamlit that allows users to simulate scenarios (e.g., heat waves or sudden drops in demand) and visualize the AI's reaction in real-time.
* **Dockerized:** Reproducible environment packaged in a Linux container for cloud deployment.

---

## 🛠️ Tech Stack

* **Core:** Python 3.9
* **Data Science:** Pandas, NumPy, Scikit-Learn.
* **AI/ML:** TensorFlow (Keras).
* **Visualization:** Matplotlib, Seaborn.
* **Deployment:** Docker + Render.

---

## 📂 Project Structure

* `app.py`: The main entry point for the Streamlit web application.
* `energy_model.h5`: The pre-trained LSTM model artifacts.
* `scaler.gz`: The fitted scaler object for normalizing/denormalizing input data.
* `Dockerfile`: Configuration for building the container image.
* `requirements.txt`: List of Python dependencies.

---

## 🚀 Local Installation (Optional)

If you want to run this project on your local machine:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Carlosmaths/energy-forecast-argentina.git]

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the App:**
    ```bash
    streamlit run app.py
    ```
Author:
Carlos Barrios: Mathematician | University Professor | Data Scientist

LinkedIn: https://www.linkedin.com/in/carlos-barrios-matematicas-fisica-machinelearning/
GitHub: https://github.com/Carlosmaths
