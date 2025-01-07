![Assistance Systems Project Banner](./img/ASP_Banner.png)

# **Chatbot Development with Rasa**

Integrating a chatbot into the **Assistance Systems Project** enhances user engagement by providing interactive, real-time assistance. This document outlines the development process of the chatbot using Rasa, detailing how the model was processed, augmented, trained, and integrated into the application.

---

## **1. Introduction to the Rasa Chatbot**

Rasa is an open-source machine learning framework for building contextual AI assistants and chatbots. It provides tools for natural language understanding (NLU) and dialogue management, allowing the creation of sophisticated conversational agents.

### **Purpose of the Chatbot**

- **Personalized Health Assistance:** Provide users with health recommendations based on their input.
- **Data Analysis Insights:** Answer queries related to data analysis findings.
- **User Engagement:** Enhance the user experience through interactive conversations.

---

## **2. Designing the Chatbot**

The chatbot's functionality is defined through several key components:

### **2.1. Intents and Entities**

- **Intents:** Represent the user's intention behind a message.
  - Examples: `greet`, `goodbye`, `ask_recommendation`, `ask_data_analysis`, `chitchat_ask_name`, `out_of_scope`.
- **Entities:** Extract specific pieces of information from user messages.
  - In this project, entities were minimally utilized due to the nature of the interactions.

### **2.2. Domain Definition**

- The `domain.yml` file specifies the chatbot's capabilities, including intents, responses, actions, and slots.
- **Actions:** Define the operations the chatbot can perform, such as providing recommendations or data analysis summaries.

### **2.3. Stories and Rules**

- **Stories (`stories.yml`):** Define example conversations to train the dialogue model.
- **Rules (`rules.yml`):** Specify deterministic paths the conversation can follow, ensuring consistent responses for certain intents.

---

## **3. Training Data and NLU**

### **3.1. NLU Training Data**

- The `nlu.yml` file contains annotated examples for each intent.
- **Example Structure:**

  ```yaml
  - intent: greet
    examples: |
      - Hi
      - Hello
      - Hey there
  ```

- **Augmentation:** Multiple examples were provided for each intent to improve the model's ability to generalize.

### **3.2. Data Augmentation**

- **Purpose:** Enhance the chatbot's understanding by exposing it to a variety of expressions.
- **Methods:**
  - **Synonym Expansion:** Added synonyms for common affirmations and denials.
  - **Paraphrasing:** Included rephrased examples for each intent.
  - **Diversification:** Used different colloquial phrases and regional expressions.

- **Impact on Model Training:**
  - Improved intent recognition accuracy.
  - Enhanced the chatbot's ability to handle varied user inputs.

---

## **4. Training the Model**

### **4.1. Model Configuration**

- The `config.yml` file defines the machine learning pipelines for NLU and dialogue management.
- **NLU Pipeline:**
  - **Tokenization:** Splitting text into tokens using `WhitespaceTokenizer`.
  - **Featurization:** Converting tokens into numerical features with `CountVectorsFeaturizer` and `LexicalSyntacticFeaturizer`.
  - **Intent Classification:** Using `DIETClassifier` for multi-intent classification.
- **Policies:**
  - **RulePolicy:** Handles conversations following defined rules.
  - **MemoizationPolicy:** Remembers exact conversation paths from training stories.
  - **TEDPolicy:** Uses machine learning to predict the next action.

### **4.2. Training Process**

- Executed the command `rasa train` to train the NLU and dialogue models.
- **Steps:**
  1. **Data Validation:** Ensured all YAML files were correctly formatted.
  2. **NLU Model Training:** The model learned to classify intents based on training examples.
  3. **Core Model Training:** The dialogue model learned conversation patterns from stories and rules.

- **Model Artifacts:**
  - Generated a model file (e.g., `models/20231010-123456.tar.gz`) containing the trained NLU and Core models.

---

## **5. Integration with the Application**

### **5.1. Connecting the Chatbot to the Web App**

- **REST API:** Utilized Rasa's REST channel to communicate between the chatbot and the Streamlit application.
- **Endpoints:**
  - **Webhook URL:** `http://rasa:5005/webhooks/rest/webhook`
  - **Status URL:** `http://rasa:5005/status`

### **5.2. Implementation in Streamlit**

- **Session Management:** Each user session is assigned a unique ID to maintain context.
- **Chat Interface:**
  - Built using Streamlit's chat components.
  - Handled user inputs and displayed chatbot responses.
- **Error Handling:**
  - Implemented checks to ensure the chatbot model is loaded and ready before processing messages.
  - Provided user-friendly messages in case of connectivity issues.

- **Code Snippet from `rasa_chatbot.py`:**

  ```python
  def send_message(self, message):
      payload = {
          "sender": self.session_id,
          "message": message,
      }
      response = requests.post(self.webhook_url, json=payload)
      return response.json()
  ```

---

## **6. Testing and Evaluation**

### **6.1. Unit Testing**

- Tested individual components such as intent classification and response retrieval.
- Ensured that custom actions executed correctly.

### **6.2. Conversation Testing**

*This section is currently backlogged and will be addressed in future updates.*

### **6.3. User Feedback**

- Collected feedback from sample users to identify misunderstandings or inappropriate responses.
- Iteratively updated training data based on feedback to improve performance.

---

## **7. Challenges and Solutions**

### **7.1. Handling Out-of-Scope Queries**

- **Issue:** Users might ask questions beyond the chatbot's knowledge base.
- **Solution:**
  - Defined an `out_of_scope` intent.
  - Added fallback responses to guide the user back to supported topics.

### **7.2. Ensuring Contextual Responses**

- **Issue:** Maintaining context over multiple turns can be challenging.
- **Solution:**
  - Utilized slots to store information throughout the conversation.
  - Implemented stories that model longer conversation paths.

### **7.3. Model Deployment**

- **Issue:** Keeping the model updated and loaded in the Rasa server.
- **Solution:**
  - Automated training and deployment using scripts was considered; however, due to current resource constraints, this is not planned.
  - Instead, manual updates and monitoring will be performed to ensure the model remains up-to-date and operational.
  - Implemented status checks before accepting user inputs in the application to verify model readiness.

---

## **8. Conclusion**

The integration of the Rasa chatbot into the **Assistance Systems Project** significantly enhances user interaction by providing personalized assistance and immediate responses to queries. By carefully designing intents, augmenting training data, and rigorously testing the model, we developed a robust chatbot that aligns with the project's objectives.

---

**Note:** For detailed implementation code and configurations, please refer to the project's repository, specifically the `chatbot` directory containing the Rasa files.

---