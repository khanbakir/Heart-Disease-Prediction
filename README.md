#  Heart Disease Prediction using Machine Learning

This project is a comprehensive analysis and predictive modeling effort to forecast the likelihood of heart disease in patients. Using **Python** and various machine learning algorithms, this project demonstrates a complete workflow from data preprocessing to model evaluation, culminating in a highly accurate predictive model.

---

## Project Objective

The primary objectives of this project were to:
* Analyze and understand the key medical attributes that contribute to heart disease.
* Perform data preprocessing and feature engineering, including creating dummy variables for categorical features and scaling numerical data using **StandardScaler**.
* Implement and train multiple machine learning classification algorithms: **Logistic Regression**, **Decision Tree Classifier**, and **K-Nearest Neighbors (KNN)**.
* Evaluate and compare the performance of each model to identify the most accurate and reliable predictor for heart disease.

---

##  Methodology and Results

The project followed a structured machine learning pipeline to ensure robust and accurate results. The key steps and final model accuracies are outlined below:

* **Data Preprocessing**: The dataset was cleaned, and categorical variables were converted into a numerical format using one-hot encoding.
* **Feature Scaling**: `StandardScaler` was applied to normalize the feature set, improving the performance of distance-based algorithms like KNN and Logistic Regression.
* **Model Training**: Three distinct classification models were trained on the preprocessed data.
* **Hyperparameter Tuning**: The Decision Tree and KNN models were tuned to find the optimal settings for maximum accuracy.

The final evaluation on the test dataset yielded the following results:

| Model                       | Test Accuracy |
| --------------------------- | :-----------: |
| Logistic Regression         |    88.52%     |
| Decision Tree (Tuned)       |      -        |
| **K-Nearest Neighbors (Tuned)** |  **91.80%** |

The **K-Nearest Neighbors (KNN)** model achieved the highest accuracy of **91.80%**, making it the most effective model for this prediction task.

---

##  Technologies Used

* **Python**: The core programming language used for the analysis.
* **Pandas**: For data manipulation and analysis.
* **Scikit-learn**: For implementing machine learning models, preprocessing, and evaluation metrics.
* **Matplotlib & Seaborn**: For data visualization and exploratory data analysis.
* **Jupyter Notebook**: As the interactive development environment.

---

