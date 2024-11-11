![Assistance Systems Project Banner](./docs/ASP_Banner.png)

# Assistance Systems Project

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
<!-- - [Demo](#demo) -->
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Application with Docker](#running-the-application-with-docker)
  - [Training the Rasa Chatbot](#training-the-rasa-chatbot)
  <!-- - [Running the Application Locally (Optional)](#running-the-application-locally-optional) -->
- [Chatbot Integration](#chatbot-integration)
- [Data](#data)
- [Modeling](#modeling)
- [Docker Setup](#docker-setup)
- [Project Structure](#project-structure)
- [License](#license)
- [Contact](#contact)

## Introduction

**Assistance Systems Project** is an advanced recommendation system designed to enhance user experience by providing personalized suggestions based on user behavior and preferences. Leveraging modern technologies such as Streamlit for the frontend and Rasa for chatbot integration, this project aims to deliver an intuitive and efficient system for diverse applications.

## Features

- **Interactive Web Interface:** Built with Streamlit, offering a seamless and responsive user experience.
- **Personalized Recommendations:** Utilizes Scikit-Learn algorithms to provide tailored suggestions.
- **Data Analysis & Visualization:** Employs Pandas and Matplotlib for insightful data analysis and visualization.
- **Chatbot Support:** Integrates a Rasa-powered chatbot to assist users and enhance interaction.
- **Robust Data Handling:** Implements strategies for outlier detection and augmentation with realistic fake data.

<!-- ## Demo -->

<!-- ![Assistance Systems Project Demo](./demo.gif) -->

<!-- Experience a live demonstration of Assistance Systems Project [here](https://your-deployment-url.com). -->

## Installation

### Prerequisites

- **Docker:** Ensure Docker is installed on your system.
- **Git:** For cloning the repository and managing submodules.

### Steps

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/project-apero.git
    cd project-apero
    ```

<!--
2. **Set Up Virtual Environments (Optional):**
    If you prefer to run the application without Docker, you can set up virtual environments for each component.

    **Create and Activate Virtual Environment for Streamlit App:**
    ```bash
    python3 -m venv venv_streamlit
    source venv_streamlit/bin/activate  # On Windows: venv_streamlit\Scripts\activate
    ```

    **Install Dependencies for Streamlit App:**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

    **Create and Activate Virtual Environment for Rasa Server:**
    Open a new terminal window for the Rasa server.
    ```bash
    python3 -m venv venv_rasa
    source venv_rasa/bin/activate  # On Windows: venv_rasa\Scripts\activate
    ```

    **Install Dependencies for Rasa Server:**
    ```bash
    pip install rasa==3.6.2
    ```

    **Create and Activate Virtual Environment for Rasa Action Server:**
    Open another terminal window for the Rasa action server.
    ```bash
    python3 -m venv venv_rasa_actions
    source venv_rasa_actions/bin/activate  # On Windows: venv_rasa_actions\Scripts\activate
    ```

    **Install Dependencies for Rasa Action Server:**
    ```bash
    pip install rasa-sdk==3.6.2
    pip install -r actions/requirements-actions.txt
    ```
-->

## Usage

### Running the Application with Docker

1. **Ensure Docker Engine is Installed:**
    Make sure Docker is installed and running on your system.

2. **Build and Start Services:**
    Navigate to the project root directory and execute:
    ```bash
    docker-compose up --build
    ```
    This command builds the Docker images and starts all services as defined in `docker-compose.yml`.

3. **Access the Streamlit Application:**
    Open your browser and navigate to [http://localhost:8501](http://localhost:8501) to access the interactive web interface.

### Training the Rasa Chatbot

If a Rasa model has not been trained in the `models/chatbot/` directory, follow these steps:

1. **Access the Rasa Server Container:**
    ```bash
    docker exec -it rasa_server bash
    ```

2. **Train the Rasa Model:**
    Inside the container, execute:
    ```bash
    rasa train
    ```

3. **Move Trained Models:**
    After training completes, move the trained model files from the `models/` directory to `models/chatbot/`:
    ```bash
    mv models/* models/chatbot/
    ```

4. **Restart Services:**
    Exit the container and rebuild the Docker services:
    ```bash
    exit
    docker-compose up --build
    ```

5. **Monitor Rasa Server Logs:**
    Ensure the Rasa server is running by checking the logs for messages like:
    ```
    2024-11-09 01:15:42 INFO     root  - Rasa server is up and running.
    2024-11-09 01:15:42 INFO     root  - Enabling coroutine debugging. Loop id 93825087865808.
    ```

6. **Finalize Setup:**
    - Navigate to the Data Analysis page in the Streamlit app and wait for the evaluation models to finish training.
    - Once evaluations are complete, models will be available for use within the Chatbot.
    - Ensure that data filters are applied as needed and that session management maintains these filters when switching between Data Analysis and Chatbot sections.

<!--
### Running the Application Locally (Optional)

If you prefer to run the application without Docker, follow these steps. Ensure that you have set up the virtual environments as described in the Installation section.

#### 1. Start the Rasa Action Server

In the terminal window with the `venv_rasa_actions` environment activated:

```bash
rasa run actions --port 5055
```

This will start the Rasa Action Server on port `5055`.

#### 2. Start the Rasa Server

In the terminal window with the `venv_rasa` environment activated:

```bash
rasa run --enable-api --cors "*" --debug --endpoints endpoints.yml
```

This will start the Rasa Server on port `5005`.

#### 3. Start the Streamlit App

In the terminal window with the `venv_streamlit` environment activated:

Ensure that the `RASA_SERVER` environment variable points to your local Rasa server:

```bash
export RASA_SERVER=http://localhost:5005/webhooks/rest/webhook
```

Then start the Streamlit app:

```bash
streamlit run src/app.py --server.port=8501 --server.address=0.0.0.0
```

This will start the Streamlit application on port `8501`.

#### 4. Access the Application

Open your browser and navigate to [http://localhost:8501](http://localhost:8501) to interact with the application.

#### 5. Training the Rasa Chatbot Locally

If a Rasa model has not been trained, you need to train it:

In the terminal window with the `venv_rasa` environment activated:

```bash
rasa train
```

This will train the Rasa model and save it in the `models` directory.

#### 6. Ensure Correct File Paths

Make sure that the file paths in your `credentials.yml`, `endpoints.yml`, and other configuration files are correctly set up to reflect the local setup.

#### 7. (Optional) Start Duckling Server Locally

If your Rasa model uses Duckling for entity extraction, you need to start the Duckling server.

In a new terminal window:

```bash
docker run -p 8000:8000 rasa/duckling
```

This will start Duckling on port `8000`.

-->

## Chatbot Integration

### Overview

The chatbot is built using the Rasa framework and is designed to interact contextually with the data analysis results presented on the Streamlit app. It can assist users in navigating the application, provide recommendations, and answer queries related to the data insights.

### Features

- **Context-Aware Conversations:** Understands the context from user interactions and provides relevant responses.
- **Data-Driven Responses:** Fetches and presents data analysis results upon user requests.
- **Seamless Integration:** Embedded within the Streamlit app for a unified user experience.

### Configuration

- **Rasa Server:** Runs on port `5005`.
- **Rasa Action Server:** Runs on port `5055`.
- **Streamlit App:** Communicates with the Rasa server via the Docker network.

### Custom Actions

Custom actions are implemented in `actions/actions.py` to enable the chatbot to fetch and present data analysis results. These actions interact with the Streamlit app's data processing modules to retrieve relevant insights based on user queries.

## Data

### Dataset

We utilize the [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) from Kaggle for training and evaluation.

### Data Handling

- **Outlier Detection:** Implemented using the Z-score method to identify and handle anomalies.
- **Data Augmentation:** Added 30% realistic synthetic data to enhance dataset robustness.
- **Data Transformation:** Normalized numerical features following best practices outlined in [Google's Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/numerical-data).

### Data Pipeline

The data pipeline is managed within the `data/` directory, ensuring organized data processing and management.

## Modeling

### Algorithms

- **Random Forest Classifier:** Chosen for its robustness and ability to handle feature interactions.
- **Support Vector Machine (SVM):** Utilized for its effectiveness in high-dimensional spaces.

### Model Training

Models are trained using Scikit-Learn, with performance evaluated based on accuracy, precision, recall, and F1-score. The best-performing model is integrated into the Streamlit app for generating personalized recommendations.

## Docker Setup

The project is containerized using Docker and orchestrated with Docker Compose to ensure consistent environments across development and production.

### Services

- **Rasa Server (`rasa_server`):** Handles natural language understanding (NLU) and dialogue management.
- **Rasa Action Server (`rasa_action_server`):** Executes custom actions defined in the project.
- **Streamlit App (`streamlit_app`):** Provides the interactive frontend for users.
- **Duckling (`duckling`):** (Optional) Extracts entities like dates, times, and numbers from user inputs.

### Running the Services

Ensure Docker and Docker Compose are installed, then execute:

```bash
docker-compose up --build
```

### Accessing Services

- **Streamlit App:** [http://localhost:8501](http://localhost:8501)
- **Rasa Server:** [http://localhost:5005](http://localhost:5005)
- **Rasa Action Server:** [http://localhost:5055](http://localhost:5055)
- **Duckling (Optional):** [http://localhost:8000](http://localhost:8000)

## Project Structure

```
project-apero/
├── actions/
│   ├── actions.py
│   ├── Dockerfile
│   ├── requirements-actions.txt
│   └── __init__.py
├── data/
│   ├── data_analysis.py
│   ├── data_augmentation.py
│   ├── data_loader.py
│   ├── data_preprocessor.py
│   ├── data_visualization.py
│   ├── nlu.yml
│   ├── processed/
│   ├── raw/
│   └── stories.yml
├── models/
│   ├── chatbot/
│   └── data_analysis/
│       └── evaluations/
├── src/
│   ├── app.py
│   ├── chatbot/
│   │   ├── rasa_chatbot.py
│   │   └── __init__.py
│   └── __init__.py
├── .dockerignore
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
└── docs/
    └── Project_Outline.md
```

## License

This project is licensed under the [GNU GENERAL PUBLIC LICENSE](./LICENSE).

## Contact

For any inquiries or support, please open an issue in the [GitHub Repository](https://github.com/hlexnc/project-apero). 
