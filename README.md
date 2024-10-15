![Project Apero Banner](./banner.png)

# Project Apero

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
<!-- - [Demo](#demo) -->
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Modeling](#modeling)
- [Chatbot Integration](#chatbot-integration)
- [Project Structure](#project-structure)
- [License](#license)
- [Contact](#contact)

## Introduction

**Project Apero** is an advanced recommendation system designed to enhance user experience by providing personalized suggestions based on user behavior and preferences. Leveraging modern technologies such as Streamlit for the frontend and Rasa for chatbot integration, this project aims to deliver an intuitive and efficient system for diverse applications.

## Features

- **Interactive Web Interface:** Built with Streamlit, offering a seamless and responsive user experience.
- **Personalized Recommendations:** Utilizes Scikit-Learn algorithms to provide tailored suggestions.
- **Data Analysis & Visualization:** Employs Pandas and Matplotlib for insightful data analysis and visualization.
- **Chatbot Support:** Integrates a Rasa-powered chatbot to assist users and enhance interaction.
- **Robust Data Handling:** Implements strategies for outlier detection and augmentation with realistic fake data.

<!-- ## Demo -->

<!-- ![Project Apero Demo](./demo.gif) -->

<!-- Experience a live demonstration of Project Apero [here](https://your-deployment-url.com). -->

## Installation

### Steps

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/project-apero.git
    cd project-apero
    ```

2. **Set Up Virtual Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install Dependencies:**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

4. **Configure Streamlit Secrets:**
    - Rename `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`.
    - Add your `OPENAI_API_KEY` and other necessary keys.

5. **Initialize Submodules:**
    ```bash
    git submodule update --init --recursive
    ```

## Usage

### Running the Streamlit App

```bash
streamlit run app.py
```

Access the application at `http://localhost:8501`.

### Interacting with the Chatbot

The integrated Rasa chatbot can be accessed within the Streamlit interface. Ensure the Rasa server is running:

```bash
cd rasa_bot
rasa run
```

## Data

### Dataset

We utilize the [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) from Kaggle for training and evaluation.

### Data Handling

- **Outlier Detection:** Implemented using Z-score method to identify and handle anomalies.
- **Data Augmentation:** Added 30% realistic synthetic data to enhance model robustness.
- **Data Transformation:** Normalized numerical features following best practices outlined in [Google's Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/numerical-data).

## Modeling

### Algorithms

- **Random Forest Classifier:** Chosen for its robustness and ability to handle feature interactions.
- **Support Vector Machine (SVM):** Utilized for its effectiveness in high-dimensional spaces.

<!-- ### Evaluation -->

<!-- Models are evaluated based on accuracy, precision, recall, and F1-score. Detailed analysis is available in the [Wiki](https://yourgitwiki-link.com). -->

## Chatbot Integration

### Justification

A chatbot enhances user interaction by providing instant support and personalized assistance, making the recommendation system more accessible and user-friendly.

### Implementation

- **Rasa Framework:** Deployed to handle natural language understanding and dialogue management.
<!-- - **Sample Dialogues:** Documented in the [Wiki](https://yourgitwiki-link.com) to demonstrate typical user interactions. -->
- **Dialog Flow:** High-level dialog flows are designed to ensure coherent and context-aware conversations.

## Project Structure

...

## License

This project is licensed under the [GNU GENERAL PUBLIC LICENSE](./LICENSE).

For any inquiries or support, please open an issue in the [GitHub Repository](https://github.com/yourusername/project-apero).

---