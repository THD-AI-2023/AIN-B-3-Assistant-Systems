![Assistance Systems Project Banner](./img/ASP_Banner.png)

# **Sample Dialogs for Chatbot Use Cases**

To demonstrate the chatbot's capabilities in the **Assistance Systems Project**, below are sample dialogs for each selected use case, including a real interaction from the running application.

---

## **Use Case 1: Personalized Health Recommendations**

### **Dialog 1**

**User:**  
Hi, can you give me a health recommendation?

**Chatbot:**  
Hello! Sure, I'd be happy to help. Please provide your age, gender, BMI, and any relevant health information.

**User:**  
I'm 45 years old, male, BMI is 28, and I have hypertension.

**Chatbot:**  
Based on your information, you may have an elevated risk of stroke. I recommend consulting a healthcare professional and focusing on maintaining a healthy lifestyle. Would you like some tips on reducing your risk?

---

### **Dialog 2**

**User:**  
Can I get a personalized recommendation?

**Chatbot:**  
Absolutely! Please share your age, gender, BMI, and any health conditions you have.

**User:**  
Age 30, female, BMI 22, no health issues.

**Chatbot:**  
Great news! Your risk of stroke appears low. Continue maintaining a healthy lifestyle. Let me know if you need any advice on health and wellness.

---

### **Dialog 3**

**User:**  
Give me a health suggestion.

**Chatbot:**  
I'd be glad to assist. Could you provide your age, gender, BMI, and any existing health conditions?

**User:**  
I'm 60, male, BMI 26, and I have heart disease.

**Chatbot:**  
Thank you. Given your information, you might be at a higher risk of stroke. It's important to follow medical advice and consider lifestyle changes. Would you like some recommendations on how to reduce your risk?

---

## **Use Case 2: Data Analysis Insights**

### **Dialog 1**

**User:**  
Can you summarize the data analysis?

**Chatbot:**  
Certainly! Our analysis shows that age, hypertension, and heart disease are significant factors influencing stroke risk. Would you like more details on any of these?

**User:**  
Yes, tell me about age.

**Chatbot:**  
Age is a major factor; as age increases, the risk of stroke tends to increase. It's important to monitor health indicators as you get older.

---

### **Dialog 2**

**User:**  
What did the data analysis find?

**Chatbot:**  
The analysis identified key factors affecting stroke risk, including age, BMI, and average glucose level. Do you want to know how these impact stroke risk?

**User:**  
Yes, explain BMI's effect.

**Chatbot:**  
A higher BMI can be associated with increased stroke risk. Maintaining a healthy weight through diet and exercise can help reduce this risk.

---

### **Dialog 3**

**User:**  
Tell me about the data insights.

**Chatbot:**  
Our data shows strong correlations between stroke risk and factors like hypertension and heart disease. Managing these conditions is crucial. Would you like advice on how to manage hypertension?

---

## **Sample Interaction from Running Application**

Below is a sample of a chatbot interaction from the running application:

```
Chatbot Assistance

Chatbot is ready to assist you.

Chatbot

Hello!

Hello! How can I assist you today?

User: Can you provide a data analysis summary?

Chatbot:

Based on our data analysis, the factors that most strongly correlate with stroke are:

- **Age** with a correlation coefficient of 0.25
- **Heart_disease** with a correlation coefficient of 0.13
- **Avg_glucose_level** with a correlation coefficient of 0.13

As age increases, the risk of stroke tends to increase. Heart disease is associated with a higher risk of stroke. Higher average glucose levels can increase stroke risk.

The best performing model is **Logistic Regression Augmented** with an accuracy of **0.86** and an F1 score of **0.26**.

User: I need a health recommendation.

Chatbot:

Based on your data (Age: 60, Gender: Male, BMI: 25.0), you may have an elevated risk of stroke. It's important to consult a healthcare professional for personalized advice.

Would you like some tips on how to reduce your stroke risk?

User: Yes please.

Chatbot:

Here are some recommendations to help reduce your stroke risk:

1. **Healthy Diet:** Eat plenty of fruits, vegetables, and whole grains.

2. **Regular Exercise:** Aim for at least 150 minutes of moderate aerobic activity each week.

3. **Avoid Smoking:** If you smoke, consider quitting.

4. **Limit Alcohol:** Drink alcohol in moderation.

5. **Monitor Health Indicators:** Keep an eye on blood pressure, cholesterol, and blood sugar levels.

6. **Stress Management:** Practice relaxation techniques like meditation or yoga.

Always consult with a healthcare professional for personalized advice.

User: Tell me a joke.

Chatbot:

Why did the computer show up at work late? It had a hard drive!

User: What's your favorite color?

Chatbot:

I like all colors in the digital spectrum.

User: Goodbye.

Chatbot:

Goodbye! Have a great day!
```

This interaction demonstrates the chatbot's ability to provide data analysis summaries, personalized health recommendations, and engage in casual conversation.

---