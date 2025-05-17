# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_imb_pipeline

# Set random seed for reproducibility
np.random.seed(42)

# Load the dataset
# Replace with your actual data source
# Example: df = pd.read_csv('customer_churn_data.csv')
print("Loading dataset...")
df = pd.read_csv('telco_churn.csv')  # Example using the Telco customer churn dataset

# Initial data exploration
print("\nData Exploration:")
print(f"Dataset shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nData types and missing values:")
print(df.info())
print("\nSummary statistics:")
print(df.describe(include='all'))

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Check class distribution
print("\nChurn distribution:")
print(df['Churn'].value_counts(normalize=True))

# Visualizations
plt.figure(figsize=(12, 6))
sns.countplot(x='Churn', data=df)
plt.title('Churn Distribution')
plt.show()

# Select numerical and categorical features
# Adjust these based on your actual dataset
target = 'Churn'
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 
                       'PhoneService', 'MultipleLines', 'InternetService',
                       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                       'TechSupport', 'StreamingTV', 'StreamingMovies',
                       'Contract', 'PaperlessBilling', 'PaymentMethod']

# Preprocessing: Handle missing values if any
# Convert TotalCharges to numeric (handling empty strings)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Feature engineering: Create new features if needed
# Example: tenure groups
df['TenureGroup'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 60, np.inf], 
                           labels=['0-1yr', '1-2yr', '2-4yr', '4-5yr', '5+yr'])
categorical_features.append('TenureGroup')

# Split data into features and target
X = df.drop(target, axis=1)
y = df[target].map({'Yes': 1, 'No': 0})  # Convert to binary

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Create preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)])

# Define models to try
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

# Train and evaluate models
results = {}
for name, model in models.items():
    # Create pipeline with SMOTE for handling class imbalance
    pipeline = make_imb_pipeline(
        preprocessor,
        SMOTE(random_state=42),
        model
    )
    
    print(f"\nTraining {name}...")
    pipeline.fit(X_train, y_train)
    
    # Predictions
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred)
    
    # Store results
    results[name] = {
        'model': pipeline,
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'report': report
    }
    
    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Compare model performance
print("\nModel Comparison:")
comparison_df = pd.DataFrame.from_dict({k: [v['accuracy'], [v['roc_auc']] for k, v in results.items()}, 
                                       orient='index', columns=['Accuracy', 'ROC AUC'])
print(comparison_df.sort_values(by='ROC AUC', ascending=False))

# Feature importance for tree-based models
for name in ['Random Forest', 'Gradient Boosting']:
    try:
        # Get feature names from one-hot encoding
        preprocessor.fit(X_train)
        feature_names = (numerical_features + 
                         list(preprocessor.named_transformers_['cat']
                             .named_steps['onehot']
                             .get_feature_names_out(categorical_features)))
        
        # Get feature importances
        importances = results[name]['model'].steps[2][1].feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Plot feature importances
        plt.figure(figsize=(12, 8))
        plt.title(f"{name} - Feature Importances")
        plt.barh(range(20), importances[indices][:20][::-1], align='center')
        plt.yticks(range(20), [feature_names[i] for i in indices[:20]][::-1])
        plt.xlabel('Relative Importance')
        plt.show()
    except Exception as e:
        print(f"Could not plot feature importances for {name}: {e}")

# Hyperparameter tuning for the best model
print("\nHyperparameter tuning for the best model...")
best_model_name = max(results.items(), key=lambda x: x[1]['roc_auc'])[0]
print(f"Tuning {best_model_name}...")

if best_model_name == 'Random Forest':
    param_grid = {
        'randomforestclassifier__n_estimators': [100, 200, 300],
        'randomforestclassifier__max_depth': [None, 10, 20, 30],
        'randomforestclassifier__min_samples_split': [2, 5, 10]
    }
elif best_model_name == 'Gradient Boosting':
    param_grid = {
        'gradientboostingclassifier__n_estimators': [100, 200],
        'gradientboostingclassifier__learning_rate': [0.05, 0.1, 0.2],
        'gradientboostingclassifier__max_depth': [3, 5, 7]
    }
else:
    param_grid = {}  # Skip tuning for other models in this example

if param_grid:
    grid_search = GridSearchCV(
        estimator=results[best_model_name]['model'],
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best ROC AUC: {grid_search.best_score_:.4f}")
    
    # Update the best model
    results[best_model_name]['model'] = grid_search.best_estimator_
    y_pred = grid_search.best_estimator_.predict(X_test)
    y_proba = grid_search.best_estimator_.predict_proba(X_test)[:, 1]
    
    print("\nImproved Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Improved ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")

# Save the best model
import joblib
best_model = results[best_model_name]['model']
joblib.dump(best_model, 'best_churn_model.pkl')
print(f"\nSaved best model ({best_model_name}) to 'best_churn_model.pkl'")

# Example prediction on new data
print("\nExample prediction:")
sample_data = X_test.iloc[:1].copy()
print("Sample customer data:")
print(sample_data)
prediction = best_model.predict_proba(sample_data)[0][1]
print(f"\nPredicted churn probability: {prediction:.2%}")
if prediction > 0.5:
    print("Prediction: Customer is likely to churn")
else:
    print("Prediction: Customer is not likely to churn")