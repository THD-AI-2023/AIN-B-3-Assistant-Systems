## Assistance Systems Project (AIN-B): Requirements Specification

### Table of Contents
1. [Introduction](#introduction)
2. [Functional Requirements](#functional-requirements)
   - [1. System Functionalities](#1-system-functionalities)
   - [2. User Stories](#2-user-stories)
   - [3. Use Cases](#3-use-cases)
   - [4. API Endpoints](#4-api-endpoints)
3. [Non-Functional Requirements](#non-functional-requirements)
4. [Requirement Documentation](#requirement-documentation)
5. [Conclusion](#conclusion)

---

### Introduction

The **Assistance Systems Project (AIN-B)** aims to develop a streamlined, data-driven recommendation system integrated with a chatbot to enhance user interaction and provide personalized suggestions. Utilizing technologies such as Streamlit for the frontend, Rasa for chatbot functionality, and Pandas for data manipulation, the project seeks to deliver an efficient and intuitive system tailored to user needs.

This Requirements Specification Document outlines the essential functional and non-functional requirements necessary to achieve the project's objectives, ensuring clarity and alignment among all stakeholders for successful and timely completion.

---

### Functional Requirements

#### 1. System Functionalities

The system must perform the following core functionalities to constitute a Minimum Viable Product (MVP):

1. **Data Import and Management**
   - **Import Data:** Allow administrators to upload datasets in predefined CSV formats.
   - **Data Preprocessing:** Handle data cleaning, including outlier detection and removal.
   - **Data Augmentation:** Add 25-50% realistic synthetic data to enhance dataset robustness.
   - **Data Transformation:** Normalize and transform data to be compatible with the application's requirements.

2. **Interactive Web Interface**
   - **Multi-Page Application:** Develop a Streamlit-based multi-page web application for seamless navigation.
   - **Input Widgets:** Implement at least three interactive input widgets to allow users to adjust feature variables.
   - **Data Visualization:** Provide visual representations of data using Pandas and Matplotlib for insights like correlations, min/max values, and medians.

3. **Recommendation Engine**
   - **Model Integration:** Utilize at least two Scikit-Learn algorithms to generate predictions based on user inputs.
   - **Model Evaluation:** Assess and document the suitability of each algorithm for the application's context.

4. **Chatbot Integration**
   - **Rasa Chatbot:** Develop a Rasa-powered chatbot to assist users with navigating the application and obtaining recommendations.
   - **Context-Aware Conversations:** Enable the chatbot to understand and utilize the context from user interactions and data analysis results.
   - **Dialog Management:** Create sample dialogs and a high-level dialog flow to ensure coherent and context-aware interactions.

5. **Documentation and Wiki Integration**
   - **Comprehensive Documentation:** Maintain detailed documentation within the MyGit Wiki, covering project descriptions, installation guides, data handling procedures, and implementation details.
   - **README Structure:** Adhere to the specified README.md structure for consistency and clarity.

6. **Video/Screencast Generation**
   - **Demonstration Video:** Produce a screencast showcasing at least three use cases, including the Rasa chatbot interaction, to demonstrate the application's functionalities.

#### 2. User Stories

**User Story 1: Data Exploration**
- *As a* user,
- *I want to* explore data visualizations,
- *so that* I can gain insights into data trends and patterns.

**User Story 2: Personalized Recommendations**
- *As a* user,
- *I want to* receive personalized recommendations based on my inputs,
- *so that* I can discover relevant suggestions tailored to my preferences.

**User Story 3: Chatbot Assistance**
- *As a* user,
- *I want to* interact with a chatbot,
- *so that* I can get instant help and navigate the application effectively.

**User Story 4: Data Upload**
- *As an* administrator,
- *I want to* upload and manage datasets,
- *so that* the system can provide accurate and relevant recommendations.

#### 3. Use Cases

**Use Case 1: Data Upload by Administrator**
- **Actors:** Administrator
- **Description:** Enables the administrator to upload and manage datasets necessary for generating recommendations.
- **Preconditions:** Administrator has access to the data upload interface.
- **Postconditions:** Data is successfully imported, cleaned, augmented, and transformed for use in the application.

**Use Case 2: Generating Recommendations**
- **Actors:** Authenticated User
- **Description:** Users receive personalized recommendations based on their input parameters adjusted via interactive widgets.
- **Preconditions:** User is interacting with the application and has adjusted input widgets.
- **Postconditions:** Recommendations are displayed to the user based on the selected input parameters.

**Use Case 3: Chatbot Interaction**
- **Actors:** User
- **Description:** Users interact with the chatbot to receive assistance in navigating the application and obtaining recommendations.
- **Preconditions:** Chatbot is active and integrated into the web interface.
- **Postconditions:** User receives relevant responses and assistance from the chatbot, enhancing their interaction with the application.

**Use Case 4: Data Visualization Exploration**
- **Actors:** User
- **Description:** Users explore various data visualizations to understand underlying data insights.
- **Preconditions:** Data has been uploaded and processed by the administrator.
- **Postconditions:** Users can view and interact with visual representations of data, such as correlations and statistical summaries.

#### 4. API Endpoints

To facilitate communication between the frontend, backend, and chatbot, the system must expose the following API endpoints:

1. **Data Management**
   - **POST /api/data/upload**
     - *Description:* Upload a new dataset.
     - *Request Body:* `multipart/form-data` with CSV file.
     - *Response:* `{ "message": "Data uploaded and processed successfully." }`

   - **GET /api/data**
     - *Description:* Retrieve available datasets.
     - *Response:* `{ "datasets": [ "dataset1.csv", "dataset2.csv", ... ] }`

2. **Recommendation Engine**
   - **GET /api/recommendations**
     - *Description:* Get personalized recommendations based on user inputs.
     - *Request Parameters:* Input parameters from interactive widgets.
     - *Response:* `{ "recommendations": [ ... ] }`

3. **Chatbot Integration**
   - **POST /api/chatbot/message**
     - *Description:* Send a message to the chatbot and receive a response.
     - *Request Body:* `{ "message": "string" }`
     - *Response:* `{ "response": "string" }`

4. **Documentation and Wiki**
   - **GET /api/wiki/content**
     - *Description:* Fetch content from the project's wiki.
     - *Response:* `{ "content": "string" }`

5. **Video/Screencast Management**
   - **POST /api/videos/upload**
     - *Description:* Upload a project demonstration video.
     - *Request Body:* `multipart/form-data` with video file.
     - *Response:* `{ "message": "Video uploaded successfully." }`

   - **GET /api/videos**
     - *Description:* Retrieve available demonstration videos.
     - *Response:* `{ "videos": [ "video1.mp4", "video2.mp4", ... ] }`

---

### Non-Functional Requirements

To ensure the system's effectiveness, usability, and reliability, the following non-functional requirements are essential:

1. **Performance**
   - The system should handle multiple users concurrently without significant performance degradation.
   - API responses should be delivered within 200ms under normal load conditions.

2. **Scalability**
   - The architecture should support horizontal scaling to accommodate increasing data volumes and user interactions.

3. **Security**
   - Implement HTTPS for all communications to ensure data security.
   - Ensure secure storage of sensitive data using industry-standard encryption techniques.
   - Protect against common vulnerabilities such as SQL injection, XSS, and CSRF.

4. **Usability**
   - The user interface should be intuitive and responsive across various devices and screen sizes.
   - Provide comprehensive documentation and in-app guidance to assist users.

5. **Reliability**
   - Ensure high availability with uptime targets of 99.9%.
   - Implement regular backups and disaster recovery plans to prevent data loss.

6. **Maintainability**
   - The codebase should follow best practices with clear documentation to facilitate maintenance and future enhancements.
   - Adopt a modular architecture to allow independent updates of different components.

7. **Compatibility**
   - Ensure compatibility with major web browsers (Chrome, Firefox, Safari, Edge).
   - Support integration with popular data sources and APIs to enhance functionality.

---

### Requirement Documentation

To ensure clarity and alignment among all stakeholders, the gathered requirements will be compiled into a formal **Requirements Specification Document**. This document encompasses:

1. **Introduction**
   - Project overview
   - Purpose of the document
   - Scope of the system

2. **Overall Description**
   - Product perspective
   - User characteristics
   - Constraints
   - Assumptions and dependencies

3. **Functional Requirements**
   - Detailed descriptions as outlined above
   - Diagrams (e.g., Use Case Diagrams) to visualize interactions

4. **Non-Functional Requirements**
   - Performance metrics
   - Security protocols
   - Usability standards

5. **System Models**
   - Architectural diagrams
   - Data flow diagrams

6. **External Interface Requirements**
   - API specifications
   - Integration points with third-party services

7. **Appendices**
   - Glossary of terms
   - Reference materials

**Tools and Formats:**
- The document will be created using Markdown for compatibility with the MyGit repository.
- Diagrams will be included as images or embedded using tools like Mermaid for dynamic rendering.

---

### Conclusion

This Requirements Specification Document serves as a comprehensive guide for the **Assistance Systems Project (AIN-B)**, outlining the essential functionalities, user interactions, and technical specifications required to achieve the project's objectives efficiently. By adhering to this specification, the development team can ensure a structured and effective approach, fostering collaboration and alignment among all stakeholders for a successful and timely project completion.

**Note:** This document is subject to updates and revisions as the project progresses. All stakeholders are encouraged to review and provide feedback to ensure the system meets the desired goals and standards.

---
