import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from scipy.stats import skew, kurtosis
from scipy.fft import fft
import pywt
import joblib

# Step 1: Streamlit UI for file upload
st.title("Machine Learning Model Training and Evaluation")
st.sidebar.header("Upload Your Dataset")

uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Data Overview:", data.head(5))

    # Drop unnecessary columns
    if 'Unnamed: 0' in data.columns:
        data.drop(['Unnamed: 0'], axis=1, inplace=True)

    # Step 2: Handling missing values
    missing_rows = data[data.isnull().any(axis=1)]
    st.write("Missing Data:", missing_rows)

    # Handle missing target column
    if data['y'].isnull().sum() > 0:
        data['y'] = data['y'].fillna(data['y'].mode()[0])

    # Step 3: Exploratory Data Analysis
    st.subheader("Class Distribution")
    sns.countplot(x='y', data=data)
    st.pyplot()

    st.subheader("Feature Correlation Heatmap")
    sns.heatmap(data.corr(), cmap='coolwarm')
    st.pyplot()

    # Step 4: Data Preprocessing
    X = data.iloc[:, :-1]  # Features
    y = data['y']          # Target

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 5: Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Model Training and Evaluation
    st.subheader("Model Training and Evaluation")

    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Support Vector Machine": SVC(random_state=42)
    }

    model_accuracies = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        model_accuracies[model_name] = accuracy

        st.subheader(f"{model_name} - Accuracy: {accuracy:.4f}")
        st.write("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        st.write("Classification Report:\n", classification_report(y_test, y_pred))

    # Step 6: Hyperparameter Tuning
    st.subheader("Hyperparameter Tuning")

    # Cross-validation setup
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Random Forest Tuning
    param_grid_rf = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    random_rf = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_distributions=param_grid_rf,
                                   n_iter=10, cv=kfold, n_jobs=-1, random_state=42)
    random_rf.fit(X_train, y_train)
    st.write("Best parameters for Random Forest:", random_rf.best_params_)
    random_rf_pred = random_rf.predict(X_test)
    accuracy = accuracy_score(y_test,random_rf_pred)
    st.write(f"Accuracy,{accuracy:.4f}")
    
    # Naive Bayes Tuning
    param_grid_nb = {'var_smoothing': np.logspace(0, -9, num=50)}
    random_nb = RandomizedSearchCV(GaussianNB(), param_distributions=param_grid_nb, n_iter=10, cv=kfold, n_jobs=-1)
    random_nb.fit(X_train, y_train)
    st.write("Best parameters for Naive Bayes:", random_nb.best_params_)
    random_nb_pred = random_nb.predict(X_test)
    accuracy = accuracy_score(y_test,random_nb_pred)
    st.write(f"Accuracy,{accuracy:.4f}")

    # KNN Tuning
    param_grid_knn = {'n_neighbors': [3, 5, 7], 'metric': ['euclidean', 'manhattan']}
    random_knn = RandomizedSearchCV(KNeighborsClassifier(), param_distributions=param_grid_knn, n_iter=10, cv=kfold,
                                    n_jobs=-1)
    random_knn.fit(X_train, y_train)
    st.write("Best parameters for KNN:", random_knn.best_params_)
    random_knn_pred = random_knn.predict(X_test)
    accuracy = accuracy_score(y_test,random_knn_pred)
    st.write(f"Accuracy,{accuracy:.4f}")

    # SVM Tuning
    param_grid_svm = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto'], 'kernel': ['linear', 'rbf']}
    random_svm = RandomizedSearchCV(SVC(), param_distributions=param_grid_svm, n_iter=10, cv=kfold, n_jobs=-1)
    random_svm.fit(X_train, y_train)
    st.write("Best parameters for SVM:", random_svm.best_params_)
    random_svm_pred = random_svm.predict(X_test)
    accuracy = accuracy_score(y_test,random_svm_pred)
    st.write(f"Accuracy,{accuracy:.4f}")

    # Save the best model
    best_model_name = max(model_accuracies, key=model_accuracies.get)
    best_model = models[best_model_name]
    joblib.dump(best_model, f"{best_model_name.replace(' ', '_')}_best_model.pkl")
    st.write(f"Best model saved as {best_model_name.replace(' ', '_')}_best_model.pkl")

    # Step 7: Model Comparison
    st.subheader("Model Comparison")
    plt.figure(figsize=(10, 6))
    plt.bar(model_accuracies.keys(), model_accuracies.values(), color='skyblue')
    plt.title("Model Comparison")
    plt.ylabel("Accuracy")
    plt.xlabel("Models")
    plt.xticks(rotation=45)
    st.pyplot()
