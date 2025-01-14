Rudaev, Alexander, 22303397

Rahman, Kazi Shafwanur, 22305619

Assistance Systems Project - Stroke Prediction Web App

https://mygit.th-deg.de/ar08397/Assistant-Systems-Project

https://mygit.th-deg.de/ar08397/Assistant-Systems-Project/-/wikis/home

[![Assistance Systems Project Banner](./docs/.ASP_Banner.png)](https://mygit.th-deg.de/ar08397/Assistant-Systems-Project)

## Project Description

The **Assistance Systems Project** is a comprehensive web application designed to help individuals assess and understand their risk of stroke. By inputting personal health metrics such as age, gender, BMI, and glucose levels, users receive personalized recommendations aimed at reducing their stroke risk. The system leverages advanced data analysis and machine learning techniques to provide actionable health advice in an intuitive and user-friendly interface.

### Key Features

1. **Personalized Stroke Risk Assessment**
   - **Input Health Metrics:** Users enter personal data including age, gender, BMI, blood glucose levels, hypertension status, and more.
   - **Risk Probability:** The application calculates a **stroke probability** between 0.0 and 1.0, offering a nuanced understanding of the user's stroke risk.
   - **Actionable Recommendations:** Based on the assessed risk, users receive tailored advice to manage and reduce their stroke risk effectively.

2. **Interactive Data Analysis and Visualization**
   - **Data Exploration:** Users can explore and filter health-related data to uncover insights and trends.
   - **Visual Insights:** The application presents data through interactive charts and heatmaps, highlighting key correlations and distributions.

3. **Conversational Chatbot Assistance**
   - **Real-Time Queries:** Engage with a built-in chatbot to ask questions about stroke risk, data insights, and receive instant recommendations.
   - **Guided Interaction:** The chatbot provides a conversational interface, making it easy for users to navigate the application and understand their health metrics.

4. **Robust Data Handling and Augmentation**
   - **Outlier Detection:** Identifies and removes anomalies in the data to ensure accurate analysis.
   - **Data Augmentation:** Enhances the dataset with synthetic data to improve the reliability of risk assessments.


## Installation

### Prerequisites

- **Docker:** Ensure Docker is installed and running on your system.
- **Git:** For cloning the repository.

### Steps to Get Started

1. **Clone the Repository:**
    ```bash
    git clone https://mygit.th-deg.de/ar08397/Assistant-Systems-Project.git
    cd Assistant-Systems-Project
    ```

2. **Build and Run Docker Services:**
    ```bash
    docker-compose up --build
    ```
    - This command builds the Docker images, trains the machine learning models, and starts all necessary services.
    - The initial setup may take several minutes as models are being trained.

3. **Access the Web Application:**
    - Open your web browser and navigate to [http://localhost:8501](http://localhost:8501).
    - You will be greeted with the home page, where you can navigate to different sections of the application.

### Training the Rasa Chatbot

If the chatbot isn't responding or you need to update its training data, follow these steps:

1. **Access the Rasa Server Container:**
    ```bash
    docker exec -it rasa_server bash
    ```

2. **Train the Rasa Model:**
    ```bash
    rasa train
    ```

3. **Restart Services:**
    ```bash
    exit
    docker-compose down
    docker-compose up --build -d
    ```

4. **Finalize Setup:**
    - Navigate to the **Data Analysis** page in the Streamlit app and wait for the model evaluations to complete.
    - Once finished, the chatbot will be fully operational and integrated with the recommendation system.

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

### Personalized Stroke Risk Assessment

1. **Navigate to Personalized Recommendations:**
    - From the sidebar, select **Personalized Recommendations**.

2. **Input Your Health Metrics:**
    - Fill in your personal health information, including age, gender, BMI, blood glucose levels, hypertension status, etc.

3. **Receive Your Stroke Probability:**
    - The system will calculate and display your **stroke probability** along with actionable recommendations to reduce your risk.

### Interactive Data Analysis

1. **Navigate to Data Analysis:**
    - Select **Data Analysis** from the sidebar.

2. **Explore and Filter Data:**
    - Use interactive widgets to filter the dataset based on various health parameters.

3. **Visualize Insights:**
    - View interactive charts and heatmaps that highlight key correlations and data distributions related to stroke risk.

### Chatbot Assistance

1. **Access the Chatbot:**
    - Click on **Chatbot** in the sidebar to open the conversational assistant.

2. **Interact with the Chatbot:**
    - Type your queries related to stroke risk, data analysis, or personalized recommendations.
    - Receive real-time, context-aware responses to help you understand and manage your health better.


## Demonstration Video

A screencast demonstrating the key functionalities of the **Assistance Systems Project**. The demo includes:

- **Application Workflow Without Rasa Model:**
  - Showcases the application's basic functionalities when the Rasa model is not trained, highlighting data analysis and recommendation features.
  
- **Training the Rasa Model:**
  - Demonstrates the steps to train the Rasa model, ensuring the chatbot is operational.
  
- **Data Analysis Workflow:**
  - Walks through the data analysis process, including loading data, preprocessing, outlier detection, and visualization.
  
- **Generating Recommendations:**
  - Illustrates how personalized health recommendations are generated based on user input and model predictions.
  
- **Chatbot Functionalities:**
  - Exhibits interactions with the Rasa chatbot, including handling user queries, providing recommendations, and managing conversations.

[Watch the Demo Video](./docs/img/ASP_Demo.mp4)

## Implementation of the Requests

The **Assistance Systems Project** encompasses a multi-faceted approach to developing a data-driven web application integrated with a chatbot for personalized health recommendations. Below is an overview of how each project request has been implemented:

1. **Multi-page Web App with Streamlit**:
    - **Home Page**:
        - **File**: `src/web/home.py`
        - **Function**: `run()`
        - **Description**: Implements the user interface for collecting personal health information through interactive forms using Streamlit's form functionalities.
    - **Data Analysis Page**:
        - **File**: `src/web/data_analysis_page.py`
        - **Function**: `run()`
        - **Description**: Handles data analysis operations, including loading, preprocessing, filtering, visualization, and model training.
    - **Personalized Recommendations Page**:
        - **File**: `src/web/recommendations.py`
        - **Function**: `run()`
        - **Description**: Displays personalized health recommendations based on user input and predictive modeling.
    - **Chatbot Page**:
        - **File**: `src/web/chatbot_page.py`
        - **Function**: `run()`
        - **Description**: Integrates the Rasa-based chatbot within the Streamlit application, enabling real-time user interactions and assistance.

2. **Data Handling and Augmentation**:
    - **Data Import**:
        - **File**: `src/data/data_loader.py`
        - **Function**: `load_data(filepath=None)`
        - **Description**: Loads the dataset from a predefined CSV file, ensuring seamless data ingestion into the application.
    - **Outlier Handling**:
        - **File**: `src/data/data_analysis.py`
        - **Function**: `preprocess_data(data)`
        - **Description**: Implements statistical methods to identify and manage outliers, enhancing data integrity and reliability.
    - **Fake Data Generation**:
        - **File**: `src/data/data_augmentation.py`
        - **Function**: `augment_data(X, y, augmentation_factor=0.3)`
        - **Description**: Utilizes the Faker library to generate synthetic data, augmenting the original dataset by 30% to improve model robustness.

3. **Machine Learning Integration with Scikit-Learn**:
    - **Model Training**:
        - **File**: `src/data/data_analysis.py`
        - **Function**: `DataAnalysis.train_models(status_text, progress_bar)`
        - **Description**: Trains multiple machine learning models, including Logistic Regression, Support Vector Machines, and Random Forest classifiers, on both real and augmented datasets.
    - **Model Evaluation**:
        - **File**: `src/data/data_analysis.py`
        - **Function**: `DataAnalysis.evaluate_model(model, model_name, X_test, y_test, data_type="")`
        - **Description**: Assesses model performance using metrics such as Accuracy, Precision, Recall, F1 Score, and ROC AUC, with results visualized within the application.
    - **Recommendation System**:
        - **File**: `src/web/recommendations.py`
        - **Function**: `run()`
        - **Description**: Generates personalized health recommendations based on user-provided data and predictive modeling outputs.

    - **Chat Recommendation System**:
        - **File**: `actions/actions.py`
        - **Function**: `ActionGenerateRecommendation.run(dispatcher, tracker, domain)`
        - **Description**: Generates personalized health recommendations based on user-provided data and predictive modeling outputs.
4. **Chatbot Development with Rasa**:
    - **Intent Recognition and Entity Extraction**:
        - **Files**: `data/nlu.yml`, `data/domain.yml`
        - **Description**: Defined intents and entities within Rasa's configuration files to enable accurate understanding of user inputs.
    - **Custom Actions**:
        - **File**: `actions/actions.py`
        - **Functions**: `ActionShowDataAnalysis.run()`, `ActionGenerateRecommendation.run()`, `ActionProvideStrokeRiskReductionAdvice.run()`, `ActionFallback.run()`
        - **Description**: Implements custom actions to handle data analysis summaries, generate recommendations, provide stroke risk reduction advice, and manage fallback responses.
    - **Integration with Streamlit**:
        - **File**: `src/web/chatbot_page.py`
        - **Function**: `run()`
        - **Description**: Ensures seamless communication between the Streamlit application and the Rasa chatbot through REST APIs.

5. **Documentation and Version Control**:
    - **MyGit Repository and Wiki**:
        - **Files**: All project files are maintained within the Git repository, with detailed documentation in the Wiki.
        - **Description**: Organizes source code, documentation, and model files in a structured manner, facilitating collaboration and version control.
    - **README Structure**:
        - **File**: `README.md`
        - **Description**: Adheres to the specified structure, providing clear instructions, project details, and comprehensive information on setup and usage.

Each component has been meticulously developed to ensure a cohesive and user-friendly application that leverages data analysis and machine learning to deliver meaningful health recommendations.

## Work Done

The **Assistance Systems Project** was developed collaboratively by two team members, each contributing distinct components to ensure a comprehensive and robust application.

### **Student 1: [Alexander Rudaev, Mat-No: 22303397]**

1. **Graphical User Interface (GUI) / Visualization**:
    - Developed the Streamlit-based multi-page web application interface.
    - Implemented interactive forms for data collection and dynamic visualizations using Altair.
    - Designed the layout and navigation structure to enhance user experience.

2. **General Data Analysis**:
    - Conducted exploratory data analysis using Pandas to uncover key insights and correlations.
    - Implemented statistical methods for outlier detection and data cleaning.
    - Integrated data visualization tools to present analysis results within the application.

3. **Sample Dialogs**:
    - Created and documented sample interaction scenarios for the Rasa chatbot.
    - Ensured that dialogues effectively cover use cases such as data analysis requests and health recommendations.
    - Collaborated on refining chatbot responses to align with user intents.

### **Student 2: [Kazi Shafwanur Rahman, Mat-No: 22305619]**

4. **Strategies for Outliers and Fake Data**:
    - Developed algorithms for identifying and managing outliers within the dataset.
    - Utilized the Faker library to generate realistic synthetic data, augmenting the original dataset by 30%.
    - Documented the approaches and their impact on model training and performance.

5. **Scikit-Learn Integration**:
    - Trained multiple machine learning models including Logistic Regression, Support Vector Machines, and Random Forest classifiers.
    - Performed model evaluation using metrics such as Accuracy, Precision, Recall, F1 Score, and ROC AUC.
    - Selected the best-performing model based on evaluation results and integrated it into the recommendation system.

6. **Dialog Flow**:
    - Designed and implemented the dialog flow within the Rasa framework to handle various user intents.
    - Configured intents, entities, and actions to support seamless interactions and accurate intent recognition.
    - Ensured that the chatbot effectively manages conversation states and provides relevant responses.

### **Both Members: Documentation and Programming**

- **Documentation**:
    - Maintained comprehensive project documentation within the MyGit Wiki, covering project setup, data handling, model training, and usage instructions.
    - Structured the README.md file according to the specified guidelines, ensuring clarity and completeness.

- **Programming**:
    - Collaborated on integrating different components of the application, including the web interface, data analysis modules, machine learning models, and chatbot functionalities.
    - Ensured code quality through consistent coding standards, thorough testing, and effective version control using Git.

This collaborative effort resulted in a well-rounded and functional application that meets the project’s objectives and provides valuable health recommendations through an intuitive user interface and intelligent chatbot assistance.

## Features

- **Interactive Web Interface:** Built with Streamlit, offering a seamless and responsive user experience.
- **Personalized Recommendations:** Utilizes Scikit-Learn algorithms to provide tailored suggestions.
- **Data Analysis & Visualization:** Employs Pandas and Altair for insightful data analysis and visualization.
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
│   │   └── ...
│   ├── raw/
│   │   └── healthcare-dataset-stroke-data.csv
│   ├── stories.yml
│   └── user_data/
│       └── ...
├── models/
│   ├── chatbot/
│   │   └── ...
│   └── data_analysis/
│       └── ...
├── src/
│   ├── app.py
│   ├── chatbot/
│   │   ├── rasa_chatbot.py
│   │   └── __init__.py
│   ├── web/
│   │   ├── home.py
│   │   ├── recommendations.py
│   │   ├── chatbot_page.py
│   │   ├── data_analysis_page.py
│   │   └── __init__.py
│   └── __init__.py
├── .dockerignore
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
└── docs/
    └── ...
```

## License

This project is licensed under the [GNU GENERAL PUBLIC LICENSE](./LICENSE).

## Contact

For any inquiries or support, please open an issue in the [MyGit Repository](https://mygit.th-deg.de/ar08397/Assistant-Systems-Project).
