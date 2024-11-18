Lastname1, Firstname1, 123456

Lastname2, Firstname2, 123456

Assistance Systems Project

https://mygit.th-deg.de/username/assistance-systems-project

https://mygit.th-deg.de/username/assistance-systems-project/wiki

![Assistance Systems Project Banner](./docs/.ASP_Banner.png)

## Project description

**Assistance Systems Project** is an advanced recommendation system designed to enhance user experience by providing personalized suggestions based on user behavior and preferences. Leveraging modern technologies such as Streamlit for the frontend and Rasa for chatbot integration, this project aims to deliver an intuitive and efficient system for diverse applications.

## Installation

### Prerequisites

- **Docker:** Ensure Docker is installed on your system.
- **Git:** For cloning the repository and managing submodules.

### Steps

1. **Clone the Repository:**
    ```bash
    git clone https://mygit.th-deg.de/yourusername/assistance-systems-project.git
    cd assistance-systems-project
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

2. **Build and Start Services:**
    ```bash
    docker-compose up --build
    ```

3. **Access the Streamlit Application:**
    Open your browser and navigate to [http://localhost:8501](http://localhost:8501) to access the interactive web interface.

> ### **IMPORTANT NOTE** :information_source:
>
> If you are using the chatbot feature for the first time, **please ensure that the Rasa model has been properly trained**. Training the model is crucial for the chatbot to function correctly. To train the Rasa model, please follow the instructions in the [Training the Rasa Chatbot](#training-the-rasa-chatbot) section below. Failing to train the Rasa model may result in unexpected behavior or errors when interacting with the chatbot.

## Data

### Dataset

We utilize the [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) from Kaggle for training and evaluation.

### Data Handling

- **Outlier Detection:** Implemented using the Z-score method to identify and handle anomalies.
- **Data Augmentation:** Added 30% realistic synthetic data to enhance dataset robustness.
- **Data Transformation:** Normalized numerical features following best practices outlined in [Google's Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/numerical-data).

## Basic Usage

### Running the Application with Docker

1. **Ensure Docker Engine is Installed:**
    Make sure Docker is installed and running on your system.

2. **Build and Start Services:**
    Navigate to the project root directory and execute:
    ```bash
    docker-compose up --build
    ```
    This command builds the Docker images, trains the Rasa model, and starts all services as defined in `docker-compose.yml`.

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

3. **Restart Services:**
    Exit the container and restart the Docker services:
    ```bash
    exit
    docker-compose up --build
    ```

4. **Monitor Rasa Server Logs:**
    Ensure the Rasa server is running by checking the logs for messages like:
    ```
    2024-11-09 01:15:42 INFO     root  - Rasa server is up and running.
    ```

5. **Finalize Setup:**
    - Navigate to the Data Analysis page in the Streamlit app and wait for the evaluation models to finish training.
    - Once evaluations are complete, models will be available for use within the Chatbot.
    - Ensure that data filters are applied as needed and that session management maintains these filters when switching between Data Analysis and Chatbot sections.

## Implementation of the Requests

*This section will be completed later, detailing how each request has been implemented and individual contributions.*

## Right-fit Question for Chatbot

*This section will provide an argument about the "right-fit" question for using a chatbot in the application. To be added later.*

## Work done

*This section will describe who has implemented each request. To be added later.*

## Features

- **Interactive Web Interface:** Built with Streamlit, offering a seamless and responsive user experience.
- **Personalized Recommendations:** Utilizes Scikit-Learn algorithms to provide tailored suggestions.
- **Data Analysis & Visualization:** Employs Pandas and Matplotlib for insightful data analysis and visualization.
- **Chatbot Support:** Integrates a Rasa-powered chatbot to assist users and enhance interaction.
- **Robust Data Handling:** Implements strategies for outlier detection and augmentation with realistic fake data.

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
assistance-systems-project/
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

For any inquiries or support, please open an issue in the [MyGit Repository](https://mygit.th-deg.de/username/assistance-systems-project).
