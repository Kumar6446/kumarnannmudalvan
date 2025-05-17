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
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_imb_pipeline

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic customer churn dataset
def generate_customer_data(num_customers=5000):
    data = {
        'CustomerID': np.arange(1, num_customers + 1),
        'Age': np.random.randint(18, 70, size=num_customers),
        'Gender': np.random.choice(['Male', 'Female'], size=num_customers),
        'Tenure': np.random.randint(1, 72, size=num_customers),  # months
        'MonthlyCharges': np.round(np.random.uniform(20, 120, size=num_customers), 2),
        'TotalCharges': np.random.uniform(50, 8000, size=num_customers),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], 
                                   size=num_customers, p=[0.5, 0.3, 0.2]),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], 
                                         size=num_customers, p=[0.4, 0.4, 0.2]),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], 
                                        size=num_customers, p=[0.3, 0.5, 0.2]),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], 
                                      size=num_customers, p=[0.3, 0.5, 0.2]),
        'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], 
                                          size=num_customers, p=[0.3, 0.5, 0.2]),
        'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 
                                        'Bank transfer', 'Credit card'], 
                                       size=num_customers),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], size=num_customers),
        'Dependents': np.random.choice(['Yes', 'No'], size=num_customers, p=[0.3, 0.7]),
        'Partner': np.random.choice(['Yes', 'No'], size=num_customers, p=[0.4, 0.6])
    }
    
    df = pd.DataFrame(data)
    
    # Calculate TotalCharges more realistically based on tenure and monthly charges
    df['TotalCharges'] = df['MonthlyCharges'] * df['Tenure'] * np.random.uniform(0.9, 1.1, size=num_customers)
    
    # Create churn label with realistic probabilities
    churn_proba = (
        0.1 + 
        (df['Contract'] == 'Month-to-month') * 0.3 +
        (df['InternetService'] == 'Fiber optic') * 0.15 +
        (df['OnlineSecurity'] == 'No') * 0.1 +
        (df['TechSupport'] == 'No') * 0.1 -
        (df['Tenure'] > 24) * 0.2 -
        (df['PaymentMethod'].isin(['Bank transfer', 'Credit card'])) * 0.1
    )
    churn_proba = np.clip(churn_proba, 0.05, 0.95)
    df['Churn'] = np.random.binomial(1, churn_proba)
    
    return df

# Generate dataset
print("Generating synthetic customer churn dataset...")
df = generate_customer_data(5000)

# Initial data exploration
print("\nData Exploration:")
print(f"Dataset shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nData types and missing values:")
print(df.info())
print("\nChurn distribution:")
print(df['Churn'].value_counts(normalize=True))

# Visualizations
plt.figure(figsize=(12, 6))
sns.countplot(x='Churn', data=df)
plt.title('Churn Distribution')
plt.show()

# Feature selection
target = 'Churn'
numerical_features = ['Age', 'Tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = ['Gender', 'Contract', 'InternetService', 'OnlineSecurity', 
                      'TechSupport', 'DeviceProtection', 'PaymentMethod',
                      'PaperlessBilling', 'Dependents', 'Partner']

# Feature engineering
df['TenureGroup'] = pd.cut(df['Tenure'], bins=[0, 12, 24, 36, 48, 60, np.inf], 
                          labels=['0-1yr', '1-2yr', '2-3yr', '3-4yr', '4-5yr', '5+yr'])
categorical_features.append('TenureGroup')

# Split data
X = df.drop([target, 'CustomerID'], axis=1)
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)])

# Models to evaluate
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# Train and evaluate models
results = {}
for name, model in models.items():
    pipeline = make_imb_pipeline(
        preprocessor,
        SMOTE(random_state=42),
        model
    )
    
    print(f"\nTraining {name}...")
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred)
    
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

# Feature importance for best model
best_model_name = max(results.items(), key=lambda x: x[1]['roc_auc'])[0]
print(f"\nBest model: {best_model_name}")

# Get feature names
preprocessor.fit(X_train)
feature_names = numerical_features + list(preprocessor.named_transformers_['cat']
                         .named_steps['onehot']
                         .get_feature_names_out(categorical_features))

# Plot feature importance
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    importances = results[best_model_name]['model'].steps[2][1].feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title(f"{best_model_name} - Feature Importances")
    plt.barh(range(20), importances[indices][:20][::-1], align='center')
    plt.yticks(range(20), [feature_names[i] for i in indices[:20]][::-1])
    plt.xlabel('Relative Importance')
    plt.show()

# Example interpretation
print("\nKey Insights from Feature Importance:")
print("1. Tenure and contract type are typically strong predictors of churn")
print("2. Customers with month-to-month contracts are more likely to churn")
print("3. Customers without tech support or online security are higher risk")
print("4. Payment methods and billing type also influence churn behavior")

# Save best model
import joblib
joblib.dump(results[best_model_name]['model'], 'churn_model.pkl')
print("\nSaved best model to 'churn_model.pkl'")

# Example prediction
sample_customer = pd.DataFrame({
    'Age': [45],
    'Gender': ['Female'],
    'Tenure': [8],
    'MonthlyCharges': [75.50],
    'TotalCharges': [600],
    'Contract': ['Month-to-month'],
    'InternetService': ['Fiber optic'],
    'OnlineSecurity': ['No'],
    'TechSupport': ['No'],
    'DeviceProtection': ['No'],
    'PaymentMethod': ['Electronic check'],
    'PaperlessBilling': ['Yes'],
    'Dependents': ['No'],
    'Partner': ['No'],
    'TenureGroup': ['0-1yr']
})

prediction = results[best_model_name]['model'].predict_proba(sample_customer)[0][1]
print(f"\nExample Prediction for High-Risk Customer:")
print(f"Churn Probability: {prediction:.1%}")
if prediction > 0.5:
    print("Action: Proactive retention offer recommended!")
else:
    print("Action: Customer appears stable")