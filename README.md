# Credit_card_default_prediction
Credit Card Default Prediction Project
Overview
This project aims to predict credit card defaults among Taiwanese credit card clients using machine learning techniques. We utilized Classification Trees and Neural Networks to build predictive models, optimize their performance through hyperparameter tuning, and evaluated them based on several performance metrics.

Table of Contents
Overview
Project Structure
Requirements
Data Preprocessing
Exploratory Data Analysis
Model Training
Hyperparameter Tuning
Evaluation
Results
Conclusion
How to Run
References
Project Structure
data/ : Contains the dataset used for the project.
notebooks/ : Jupyter notebook with detailed code and explanations.
images/ : Contains visualizations generated during the EDA phase.
README.md : Project documentation.
Requirements
Python 3.7+
Jupyter Notebook
Pandas
NumPy
Scikit-learn
Keras
Matplotlib
Seaborn
You can install the required packages using:

bash
Copy code
pip install -r requirements.txt
Data Preprocessing
Data preprocessing involves cleaning the dataset by handling missing values, encoding categorical variables, and scaling numerical features. This ensures the data is in a suitable format for training machine learning models.

Exploratory Data Analysis
Exploratory Data Analysis (EDA) is performed to understand the data distribution and relationships between variables. We use visualizations like histograms, box plots, and heatmaps to uncover patterns and insights.

Model Training
We trained two models:

Classification Tree: A decision tree classifier is used to predict defaults.
Neural Network (Multilayer Perceptron): A neural network with multiple hidden layers is implemented using Keras.
Hyperparameter Tuning
Hyperparameter tuning is conducted using GridSearchCV to find the best parameters for both models. This process helps in improving the performance by selecting optimal values for model parameters.

Evaluation
Models are evaluated based on the following metrics:

Accuracy: Proportion of correctly predicted instances.
Error Rate: Proportion of incorrectly predicted instances.
ROC AUC: Area under the ROC curve, indicating the ability of the model to distinguish between classes.
R-Squared: Proportion of variance explained by the model.
Results
Model	Accuracy	Error Rate	ROC AUC	R-Squared
Decision Tree	0.781	0.219	0.509	0.781
Tuned Decision Tree	0.819	0.181	0.753	0.819
Neural Network (MLP)	0.781	0.219	0.595	0.781
Conclusion
The Tuned Decision Tree model performed better than the base Decision Tree and Neural Network models in terms of accuracy and ROC AUC. Hyperparameter tuning significantly improved the decision tree model's performance, making it more suitable for this dataset.

How to Run
Clone the repository:
bash
Copy code
git clone https://github.com/yourusername/credit-card-default-prediction.git
Navigate to the project directory:
bash
Copy code
cd credit-card-default-prediction
Install the required packages:
bash
Copy code
pip install -r requirements.txt
Open the Jupyter notebook and run the cells:
bash
Copy code
jupyter notebook notebooks/credit_card_default_prediction.ipynb
References
UCI Machine Learning Repository: Default of Credit Card Clients Dataset link
Scikit-learn documentation link
Keras documentation link
Feel free to reach out if you have any questions or suggestions regarding the project. Happy coding! 😊
