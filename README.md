# Hypertension Prediction using Decision Tree

## Project Description
This project aims to build a Machine Learning model based on the **Decision Tree Classifier** algorithm to predict the likelihood of an individual having hypertension. The model is trained using the `hypertension_dataset.csv` dataset, which contains various risk factors such as blood pressure history, diet, family history, physical activity, and smoking habits.

## Dataset
The dataset used is `hypertension_dataset.csv`, containing the following features:
- **Salt_Intake**: Salt consumption level (low, medium, high)
- **BP_History**: Blood pressure history
- **Medication**: Medication usage
- **Family_History**: Family history of hypertension
- **Exercise_Level**: Physical activity level
- **Smoking_Status**: Smoking status
- **Has_Hypertension**: Target label (Yes/No)

**Data Source:**  
Miadul. (2023). *Hypertension Risk Prediction Dataset*. Kaggle.  
[https://www.kaggle.com/datasets/miadul/hypertension-risk-prediction-dataset](https://www.kaggle.com/datasets/miadul/hypertension-risk-prediction-dataset)

## Workflow
1. **Import Libraries**  
   Using Python libraries such as:
   - Pandas & NumPy for data manipulation
   - Scikit-learn for preprocessing, modeling, and evaluation
   - Matplotlib & Seaborn for visualization

2. **Data Cleaning & Encoding**  
   - Transforming categorical variables into numeric values using `LabelEncoder`
   - Ensuring the data is clean and free from missing values (NaN)

3. **Data Splitting**  
   - Splitting the dataset into **train** and **test** sets using `train_test_split`

4. **Model Training**  
   - Using `DecisionTreeClassifier`
   - Performing hyperparameter tuning with `GridSearchCV` to find the best parameters

5. **Model Evaluation**  
   - Using **Confusion Matrix**
   - **Classification Report** (Precision, Recall, F1-score)
   - Model accuracy (`accuracy_score`)
   - Decision tree visualization using `plot_tree`

## Evaluation Results
- **Accuracy**: Shown using `accuracy_score`
- **Precision, Recall, F1-score**: Shown using `classification_report`
- **Visualization**: Decision tree diagram and confusion matrix heatmap

## How to Run the Project
1. Make sure Python 3.x is installed
2. Install the dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
3. Run the notebook:
   ```bash
   jupyter notebook filename.ipynb
   ```
4. Ensure that the file `hypertension_dataset.csv` is available in the same directory

## Project Structure
```
.
├── notebook.ipynb               # Main notebook
└── README.md                    # Documentation
```

## License
This project is created for educational purposes. Feel free to use and modify it as needed.
