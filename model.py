import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import joblib
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# Generate a larger dataset for training with more realistic and complex patterns
def generate_training_data(n_samples=10000, fraud_ratio=0.05):
    np.random.seed(42)  # For reproducibility

    # Generate legitimate transactions
    n_legit = int(n_samples * (1 - fraud_ratio))
    legit_data = {
        'amount': np.random.lognormal(4, 1, n_legit),
        'merchant_category': np.random.choice(['grocery', 'restaurant', 'entertainment', 'travel', 'online', 'retail', 'other'], n_legit),
        'hour_of_day': np.random.randint(0, 24, n_legit),  # More realistic distribution across all hours
        'recent_transactions': np.random.randint(0, 20, n_legit),
        'is_fraud': np.zeros(n_legit, dtype=int)
    }

    # Add noise to legitimate transactions to make the problem harder
    legit_outliers = int(n_legit * 0.05)  # 5% of legitimate transactions will have "suspicious" patterns
    outlier_indices = np.random.choice(n_legit, legit_outliers, replace=False)

    # Make some legitimate transactions look suspicious
    for idx in outlier_indices:
        # Random choice of suspicious pattern
        pattern_type = np.random.randint(0, 3)
        if pattern_type == 0:
            # Unusual hour
            legit_data['hour_of_day'][idx] = np.random.choice([0, 1, 2, 3, 4, 22, 23])
        elif pattern_type == 1:
            # Unusual amount
            if np.random.random() < 0.5:
                legit_data['amount'][idx] = np.random.uniform(1, 10)
            else:
                legit_data['amount'][idx] = np.random.uniform(900, 5000)
        elif pattern_type == 2:
            # Few recent transactions
            legit_data['recent_transactions'][idx] = np.random.randint(0, 3)

    # Generate fraudulent transactions with more varied patterns
    n_fraud = int(n_samples * fraud_ratio)

    # Create different fraud profiles
    fraud_profiles = np.random.choice(['test_then_large', 'late_night', 'inactive_account', 'mixed'],
                                     n_fraud, p=[0.3, 0.3, 0.3, 0.1])

    # Initialize fraud data arrays
    fraud_amounts = np.zeros(n_fraud)
    fraud_hours = np.zeros(n_fraud, dtype=int)
    fraud_recent_txns = np.zeros(n_fraud, dtype=int)

    # Generate data based on fraud profiles
    for i, profile in enumerate(fraud_profiles):
        if profile == 'test_then_large':
            fraud_amounts[i] = np.random.choice([np.random.uniform(1, 10), np.random.uniform(900, 5000)])
            fraud_hours[i] = np.random.randint(0, 24)  # Any hour for this profile
            fraud_recent_txns[i] = np.random.randint(0, 20)  # Any number of recent transactions
        elif profile == 'late_night':
            fraud_amounts[i] = np.random.lognormal(4, 1)  # Normal amount distribution
            fraud_hours[i] = np.random.choice([0, 1, 2, 3, 4, 22, 23])  # Late night hours
            fraud_recent_txns[i] = np.random.randint(0, 20)  # Any number of recent transactions
        elif profile == 'inactive_account':
            fraud_amounts[i] = np.random.lognormal(4, 1)  # Normal amount distribution
            fraud_hours[i] = np.random.randint(0, 24)  # Any hour
            fraud_recent_txns[i] = np.random.randint(0, 3)  # Few recent transactions
        else:  # mixed profile - completely random to introduce noise
            fraud_amounts[i] = np.random.lognormal(4, 1)  # Normal amount distribution
            fraud_hours[i] = np.random.randint(0, 24)  # Any hour
            fraud_recent_txns[i] = np.random.randint(0, 20)  # Any number of recent transactions

    fraud_data = {
        'amount': fraud_amounts,
        'merchant_category': np.random.choice(['grocery', 'restaurant', 'entertainment', 'travel', 'online', 'retail', 'other'], n_fraud),
        'hour_of_day': fraud_hours,
        'recent_transactions': fraud_recent_txns,
        'is_fraud': np.ones(n_fraud, dtype=int)
    }

    # Add noise to fraud data
    noisy_fraud = int(n_fraud * 0.1)  # 10% of fraud transactions will have atypical patterns
    noisy_indices = np.random.choice(n_fraud, noisy_fraud, replace=False)

    for idx in noisy_indices:
        # Randomize all features for these transactions
        fraud_data['amount'][idx] = np.random.lognormal(4, 1)
        fraud_data['hour_of_day'][idx] = np.random.randint(9, 17)  # Normal business hours
        fraud_data['recent_transactions'][idx] = np.random.randint(5, 20)  # Normal activity level

    # Combine and shuffle
    df_legit = pd.DataFrame(legit_data)
    df_fraud = pd.DataFrame(fraud_data)
    df = pd.concat([df_legit, df_fraud], ignore_index=True)
    df = df.sample(frac=1).reset_index(drop=True)  # Shuffle

    # Convert categorical data to numeric
    df = pd.get_dummies(df, columns=['merchant_category'], drop_first=True)

    # Add a few additional features to increase complexity
    df['amount_squared'] = df['amount'] ** 2
    df['hour_category'] = pd.cut(df['hour_of_day'],
                                 bins=[0, 6, 12, 18, 24],
                                 labels=['night', 'morning', 'afternoon', 'evening']).astype('object')
    df = pd.get_dummies(df, columns=['hour_category'], drop_first=True)

    return df

def train_model():
    # Generate more complex training data
    print("Generating training data with complex patterns...")
    df = generate_training_data(n_samples=50000, fraud_ratio=0.05)

    # Split features and target
    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']

    # Create a separate validation set
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


    # Create a scaler
    scaler = StandardScaler()

    # Define resampling strategy for imbalanced data
    # Use SMOTE to oversample the minority class, then undersample the majority class to balance
    oversample = SMOTE(sampling_strategy=0.1, random_state=42)  # Increase minority class to 10% of majority
    undersample = RandomUnderSampler(sampling_strategy=0.5, random_state=42)  # Balance classes after SMOTE

    # Create a Random Forest with more conservative parameters to reduce overfitting
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,  # Limit tree depth to prevent overfitting
        min_samples_split=5,  # Require more samples to split a node
        min_samples_leaf=5,   # Require more samples in leaf nodes
        max_features='sqrt',  # Use square root of total features for each tree
        bootstrap=True,       # Sample with replacement
        class_weight='balanced_subsample',  # Account for class imbalance
        random_state=42
    )

    # Cross-validation to evaluate model
    print("Evaluating model with cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, scaler.fit_transform(X_train), y_train, cv=cv, scoring='roc_auc')
    print(f"Cross-validated ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Create a pipeline with preprocessing and resampling
    pipeline = ImbPipeline([
        ('scaler', scaler),
        ('oversample', oversample),
        ('undersample', undersample),
        ('classifier', model)
    ])

    # Train the pipeline
    print("Training model with resampling...")
    pipeline.fit(X_train, y_train)

    # Evaluate on validation set
    y_pred = pipeline.predict(X_val)
    y_pred_proba = pipeline.predict_proba(X_val)[:, 1]

    accuracy = accuracy_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_pred_proba)

    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation ROC-AUC: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_pred))

    # Save the pipeline (includes preprocessing and model)
    print("Saving pipeline and feature names...")
    joblib.dump(pipeline, 'fraud_detection_pipeline.joblib')
    joblib.dump(X.columns.tolist(), 'feature_names.joblib')

    return pipeline

if __name__ == "__main__":
    train_model()