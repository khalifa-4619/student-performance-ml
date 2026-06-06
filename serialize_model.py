import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge

def main():
    print("⏳ Starting model serialization pipeline...")
    
    # 1. Load and clean raw data
    data_path = 'data/student-mat.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing data file at {data_path}. Please place your dataset there.")
        
    df = pd.read_csv(data_path, sep=';')
    df_filtered = df[df['G3'] > 0].copy()

    # 2. Separate features (No midterms allowed)
    X = df_filtered.drop(columns=['G1', 'G2', 'G3'])
    y = df_filtered['G3']

    # 3. Dynamic typing identification
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    # 4. Construct Transformer Component
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ]
    )

    # 5. Build an atomic combined Pipeline
    # This binds the transformation logic directly onto the model execution logic
    from sklearn.pipeline import Pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', Ridge(alpha=5.0))
    ])

    # 6. Execute Leak-proof splitting and fit the pipeline
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("🏋️  Training model pipeline...")
    model_pipeline.fit(X_train, y_train)

    # 7. Serialize the entire pipeline object to disk
    model_output_dir = 'backend/app/models'
    os.makedirs(model_output_dir, exist_ok=True)
    model_output_path = os.path.join(model_output_dir, 'behavioral_ridge_v1.joblib')
    
    joblib.dump(model_pipeline, model_output_path)
    print(f"📦 Successfully serialized whole pipeline to: {model_output_path}")

if __name__ == "__main__":
    main()