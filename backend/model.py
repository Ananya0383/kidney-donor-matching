import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle
import os

# Load the donor dataset
donor_data = pd.read_csv('backend/donor list.csv')

# Preprocess the data for decision tree training
def preprocess_data(df, label_encoders=None, fit_encoders=True):
    # Initialize label_encoders if not provided
    if label_encoders is None:
        label_encoders = {}
    
    # Columns to encode
    categorical_columns = ['Blood_type', 'Region', 'sex']
    
    for col in categorical_columns:
        if fit_encoders:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        else:
            le = label_encoders[col]
            df[col + '_encoded'] = le.transform(df[col].astype(str))
    
    # Target variable encoding
    if fit_encoders:
        le_target = LabelEncoder()
        df['donor_still_on_list_encoded'] = le_target.fit_transform(df['donor_still_on_list'])
        label_encoders['donor_still_on_list'] = le_target
    else:
        le_target = label_encoders['donor_still_on_list']
        df['donor_still_on_list_encoded'] = le_target.transform(df['donor_still_on_list'])
    
    # Features and target for training
    features = df[['age_at_list_registration', 'Blood_type_encoded', 'Region_encoded', 'sex_encoded']]
    target = df['donor_still_on_list_encoded']
    
    return features, target, label_encoders

# Train decision tree model and evaluate accuracy
def train_decision_tree(df):
    print("Starting model training...")
    features, target, label_encoders = preprocess_data(df, fit_encoders=True)
    
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
    
    # Train the Decision Tree Classifier
    decision_tree_model = DecisionTreeClassifier(random_state=42)
    decision_tree_model.fit(X_train, y_train)
    
    # Save the model and label encoders for future use
    with open('backend/decision_tree_model.pkl', 'wb') as file:
        pickle.dump(decision_tree_model, file)
    with open('backend/label_encoders.pkl', 'wb') as file:
        pickle.dump(label_encoders, file)
    print("Model and label encoders trained and saved.")
    
    # Predict on the test set
    y_pred = decision_tree_model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of the Decision Tree model: {accuracy * 100:.2f}%")
    
    return decision_tree_model, label_encoders

# Try to load an existing model and label encoders, or train a new one
try:
    with open('backend/decision_tree_model.pkl', 'rb') as file:
        decision_tree_model = pickle.load(file)
    with open('backend/label_encoders.pkl', 'rb') as file:
        label_encoders = pickle.load(file)
    print("Model loaded from saved file.")
    
    # Evaluate the loaded model on the entire dataset
    features, target, _ = preprocess_data(donor_data, label_encoders=label_encoders, fit_encoders=False)
    y_pred = decision_tree_model.predict(features)
    accuracy = accuracy_score(target, y_pred)
    print(f"Accuracy of the loaded Decision Tree model on the entire dataset: {accuracy * 100:.2f}%")
except (FileNotFoundError, IOError):
    print("Model not found, training a new model.")
    decision_tree_model, label_encoders = train_decision_tree(donor_data)

# Use the decision tree to find the best donor
def find_best_donor_with_decision_tree(donor_df, tissue_type, medical_history, demographics, age, sex, weight, height, waiting_time, urgency):
    # Filter based on blood type, region, and sex
    potential_donors = donor_df[
        (donor_df['Blood_type'] == tissue_type) &
        (donor_df['Region'] == demographics) &
        (donor_df['sex'] == sex)
    ]
    
    # If no potential donors, return None
    if potential_donors.empty:
        return None
    
    # Preprocess potential donors for prediction using loaded label encoders
    potential_donors = potential_donors.copy()
    features, _, _ = preprocess_data(potential_donors, label_encoders=label_encoders, fit_encoders=False)
    
    # Predict using the decision tree model
    predictions = decision_tree_model.predict(features)
    potential_donors['still_on_list_pred'] = predictions
    
    # Filter only those predicted to be still on the list
    potential_donors = potential_donors[potential_donors['still_on_list_pred'] == 1]
    
    # If no potential donors are found after decision tree prediction, return None
    if potential_donors.empty:
        return None
    
    # Initialize donor scores
    donor_scores = []
    
    # Custom scoring based on the original logic
    for idx, donor in potential_donors.iterrows():
        score = 0
        
        # Age Difference Score
        age_difference = abs(age - int(donor['age_at_list_registration']))
        if age_difference <= 10:
            score += 10
        elif age_difference <= 20:
            score += 5
        else:
            score += 1

        # Urgency Score
        if urgency.lower() == 'high':
            score += 20
        elif urgency.lower() == 'medium':
            score += 10
        else:
            score += 5

        # HLA Matching Score (adjust this part based on your matching logic)
        hla_match_count = 0
        if donor['HLA_A1'] == tissue_type:
            hla_match_count += 1
        if donor['HLA_A2'] == tissue_type:
            hla_match_count += 1
        score += hla_match_count * 15

        # Penalize for certain conditions
        if donor['underline_disease'] == medical_history:
            score -= 10
        if donor['gestation'] == 'Yes':
            score -= 15
        if donor['prior_transplant'] == 'Yes':
            score -= 5

        # Append score and donor data
        donor_scores.append((score, donor))
    
    # Sort by score (highest score first)
    donor_scores.sort(key=lambda x: x[0], reverse=True)
    
    # Return the donor with the highest score
    return donor_scores[0][1] if donor_scores else None
