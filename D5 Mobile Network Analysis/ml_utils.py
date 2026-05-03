import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, label_binarize
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, 
    precision_score, recall_score, f1_score, accuracy_score, 
    classification_report, confusion_matrix, roc_curve, auc
)
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
import joblib
import logging
import base64
from io import BytesIO

# Set up directories
MODEL_DIR = 'models'
RESULTS_DIR = 'results'

# Ensure matplotlib doesn't use GUI backend
import matplotlib
matplotlib.use('Agg')

def plot_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buffer = BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight', dpi=150, facecolor='white')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close(fig)
    
    graphic = base64.b64encode(image_png)
    graphic = graphic.decode('utf-8')
    return graphic

def load_and_preprocess_data(df):
    """Load and preprocess the dataset"""
    try:
        # Make a copy to avoid modifying original
        df_processed = df.copy()
        
        # Handle signal strength - convert to absolute and normalize
        if 'Signal Strength (dBm)' in df_processed.columns:
            df_processed['Signal Strength (dBm)'] = np.abs(df_processed['Signal Strength (dBm)'])
            scaler = MinMaxScaler()
            df_processed['Signal Strength (dBm)'] = scaler.fit_transform(
                df_processed[['Signal Strength (dBm)']]
            )
        
        # Drop unnamed columns
        df_processed = df_processed.loc[:, ~df_processed.columns.str.contains('^Unnamed')]
        
        # Encode categorical variables
        label_encoders = {}
        for col in df_processed.select_dtypes(include='object').columns:
            if col not in ['Network Type']:  # Don't encode target for now
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                label_encoders[col] = le
        
        # Handle Network Type separately
        if 'Network Type' in df_processed.columns:
            le_target = LabelEncoder()
            df_processed['Network Type'] = le_target.fit_transform(df_processed['Network Type'].astype(str))
            label_encoders['Network Type'] = le_target
        
        # Fill missing values
        df_processed = df_processed.fillna(df_processed.mean(numeric_only=True))
        
        return df_processed, label_encoders
    
    except Exception as e:
        logging.error(f"Error in preprocessing: {str(e)}")
        raise

def create_eda_plots(df):
    """Create EDA plots and return as base64 encoded images"""
    plots = {}
    
    try:
        # Data preprocessing
        df_processed, _ = load_and_preprocess_data(df)
        
        # 1. Dataset Overview
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Dataset Overview', fontsize=16, fontweight='bold')
        
        # Shape info
        axes[0, 0].text(0.1, 0.7, f'Dataset Shape: {df.shape}', fontsize=14, fontweight='bold')
        axes[0, 0].text(0.1, 0.5, f'Number of Features: {df.shape[1]}', fontsize=12)
        axes[0, 0].text(0.1, 0.3, f'Number of Records: {df.shape[0]}', fontsize=12)
        axes[0, 0].set_xlim(0, 1)
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].axis('off')
        axes[0, 0].set_title('Basic Information')
        
        # Missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            missing_values[missing_values > 0].plot(kind='bar', ax=axes[0, 1])
            axes[0, 1].set_title('Missing Values by Column')
            axes[0, 1].set_ylabel('Count')
        else:
            axes[0, 1].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', fontsize=14)
            axes[0, 1].set_xlim(0, 1)
            axes[0, 1].set_ylim(0, 1)
            axes[0, 1].axis('off')
            axes[0, 1].set_title('Missing Values')
        
        # Data types
        dtype_counts = df.dtypes.value_counts()
        axes[1, 0].pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%')
        axes[1, 0].set_title('Data Types Distribution')
        
        # Network Type distribution
        if 'Network Type' in df.columns:
            network_counts = df['Network Type'].value_counts()
            axes[1, 1].bar(network_counts.index, network_counts.values)
            axes[1, 1].set_title('Network Type Distribution')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].tick_params(axis='x', rotation=45)
        else:
            axes[1, 1].text(0.5, 0.5, 'Network Type\nNot Available', ha='center', va='center', fontsize=14)
            axes[1, 1].set_xlim(0, 1)
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].axis('off')
            axes[1, 1].set_title('Network Type Distribution')
        
        plt.tight_layout()
        plots['overview'] = plot_to_base64(fig)   # ✅ always pass fig
        
        # 2. Signal Strength Distribution
        if 'Signal Strength (dBm)' in df.columns:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('Signal Strength Analysis', fontsize=16, fontweight='bold')
            
            # Original signal strength histogram
            axes[0].hist(df['Signal Strength (dBm)'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0].set_title('Signal Strength Distribution (Original)')
            axes[0].set_xlabel('Signal Strength (dBm)')
            axes[0].set_ylabel('Frequency')
            
            # Box plot by network type
            if 'Network Type' in df.columns:
                df.boxplot(column='Signal Strength (dBm)', by='Network Type', ax=axes[1])
                axes[1].set_title('Signal Strength by Network Type')
                axes[1].set_xlabel('Network Type')
                axes[1].set_ylabel('Signal Strength (dBm)')
            else:
                axes[1].text(0.5, 0.5, 'Network Type\nNot Available', ha='center', va='center', fontsize=14)
                axes[1].set_xlim(0, 1)
                axes[1].set_ylim(0, 1)
                axes[1].axis('off')
            
            plt.tight_layout()
            plots['signal_strength'] = plot_to_base64(fig)
        
        # 3. Correlation Matrix
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            fig, ax = plt.subplots(figsize=(12, 10))
            correlation_matrix = df_processed[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, ax=ax, fmt='.2f')
            ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plots['correlation'] = plot_to_base64(fig)
        
        # 4. Feature Distributions
        numeric_features = ['Data Throughput (Mbps)', 'Latency (ms)', 'Signal Quality (%)']
        available_features = [col for col in numeric_features if col in df.columns]
        
        if available_features:
            n_features = len(available_features)
            n_cols = min(3, n_features)
            n_rows = (n_features + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            axes = np.atleast_1d(axes).flatten()   # ✅ always flatten axes to array
            
            fig.suptitle('Feature Distributions', fontsize=16, fontweight='bold')
            
            for i, feature in enumerate(available_features):
                ax = axes[i]
                df[feature].hist(bins=30, alpha=0.7, ax=ax, color='lightgreen', edgecolor='black')
                ax.set_title(f'{feature} Distribution')
                ax.set_xlabel(feature)
                ax.set_ylabel('Frequency')
            
            # Hide empty subplots
            for j in range(len(available_features), len(axes)):
                axes[j].axis('off')
            
            plt.tight_layout()
            plots['distributions'] = plot_to_base64(fig)   # ✅ always pass fig
        
        return plots
        
    except Exception as e:
        logging.error(f"Error creating EDA plots: {str(e)}")
        return {'error': str(e)}


def train_ridge_models(X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test):
    """Train Ridge classifier and regressor"""
    results = {}
    models = {}
    
    try:
        # Ridge Classifier
        ridge_classifier = RidgeClassifier(random_state=42)
        ridge_classifier.fit(X_train, y_class_train)
        
        # Ridge Regressor  
        ridge_regressor = Ridge(random_state=42)
        ridge_regressor.fit(X_train, y_reg_train)
        
        # Predictions
        y_class_pred = ridge_classifier.predict(X_test)
        y_reg_pred = ridge_regressor.predict(X_test)
        
        # Classification metrics
        class_metrics = {
            'accuracy': accuracy_score(y_class_test, y_class_pred) * 100,
            'precision': precision_score(y_class_test, y_class_pred, average='macro') * 100,
            'recall': recall_score(y_class_test, y_class_pred, average='macro') * 100,
            'f1_score': f1_score(y_class_test, y_class_pred, average='macro') * 100
        }
        
        # Regression metrics
        reg_metrics = {
            'mae': mean_absolute_error(y_reg_test, y_reg_pred),
            'mse': mean_squared_error(y_reg_test, y_reg_pred),
            'rmse': np.sqrt(mean_squared_error(y_reg_test, y_reg_pred)),
            'r2': r2_score(y_reg_test, y_reg_pred)
        }
        
        # Save models
        joblib.dump(ridge_classifier, os.path.join(MODEL_DIR, 'ridge_classifier.pkl'))
        joblib.dump(ridge_regressor, os.path.join(MODEL_DIR, 'ridge_regressor.pkl'))
        
        models['Ridge'] = {
            'classifier': ridge_classifier,
            'regressor': ridge_regressor
        }
        
        results['Ridge'] = {
            'classification': class_metrics,
            'regression': reg_metrics
        }
        
        return models, results
        
    except Exception as e:
        logging.error(f"Error training Ridge models: {str(e)}")
        raise

def train_decision_tree_models(X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test):
    """Train Decision Tree classifier and regressor"""
    results = {}
    models = {}
    
    try:
        # Decision Tree Classifier

        dt_classifier = DecisionTreeClassifier(
            random_state=42,
            max_depth=3,        # Limit tree depth
            min_samples_split=10, # Minimum samples to split an internal node
            min_samples_leaf=5,   # Minimum samples at leaf
            max_features=0.5,     # Use only a fraction of features
        )

        dt_classifier.fit(X_train, y_class_train)
        
        # Decision Tree Regressor
        dt_regressor = DecisionTreeRegressor(random_state=42)
        dt_regressor.fit(X_train, y_reg_train)
        
        # Predictions
        y_class_pred = dt_classifier.predict(X_test)
        y_reg_pred = dt_regressor.predict(X_test)
        
        # Classification metrics
        class_metrics = {
            'accuracy': accuracy_score(y_class_test, y_class_pred) * 100,
            'precision': precision_score(y_class_test, y_class_pred, average='macro') * 100,
            'recall': recall_score(y_class_test, y_class_pred, average='macro') * 100,
            'f1_score': f1_score(y_class_test, y_class_pred, average='macro') * 100
        }
        
        # Regression metrics
        reg_metrics = {
            'mae': mean_absolute_error(y_reg_test, y_reg_pred),
            'mse': mean_squared_error(y_reg_test, y_reg_pred),
            'rmse': np.sqrt(mean_squared_error(y_reg_test, y_reg_pred)),
            'r2': r2_score(y_reg_test, y_reg_pred)
        }
        
        # Save models
        joblib.dump(dt_classifier, os.path.join(MODEL_DIR, 'decision_tree_classifier.pkl'))
        joblib.dump(dt_regressor, os.path.join(MODEL_DIR, 'decision_tree_regressor.pkl'))
        
        models['Decision Tree'] = {
            'classifier': dt_classifier,
            'regressor': dt_regressor
        }
        
        results['Decision Tree'] = {
            'classification': class_metrics,
            'regression': reg_metrics
        }
        
        return models, results
        
    except Exception as e:
        logging.error(f"Error training Decision Tree models: {str(e)}")
        raise

def train_hybrid_mlp_catboost(X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test):
    """Train hybrid MLP-CatBoost models"""
    results = {}
    models = {}
    
    try:
        # Import CatBoost
        from catboost import CatBoostClassifier, CatBoostRegressor
        
        # For classification: Use CatBoost as primary
        catboost_classifier = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6, verbose=False, random_state=42)
        catboost_classifier.fit(X_train, y_class_train)
        
        # For regression: Use CatBoost as primary  
        catboost_regressor = CatBoostRegressor(iterations=100, learning_rate=0.1, depth=6, verbose=False, random_state=42)
        catboost_regressor.fit(X_train, y_reg_train)
        
        # Predictions
        y_class_pred = catboost_classifier.predict(X_test)
        y_reg_pred = catboost_regressor.predict(X_test)
        
        # Classification metrics
        class_metrics = {
            'accuracy': accuracy_score(y_class_test, y_class_pred) * 100,
            'precision': precision_score(y_class_test, y_class_pred, average='macro') * 100,
            'recall': recall_score(y_class_test, y_class_pred, average='macro') * 100,
            'f1_score': f1_score(y_class_test, y_class_pred, average='macro') * 100
        }
        
        # Regression metrics
        reg_metrics = {
            'mae': mean_absolute_error(y_reg_test, y_reg_pred),
            'mse': mean_squared_error(y_reg_test, y_reg_pred),
            'rmse': np.sqrt(mean_squared_error(y_reg_test, y_reg_pred)),
            'r2': r2_score(y_reg_test, y_reg_pred)
        }
        
        # Save models
        joblib.dump(catboost_classifier, os.path.join(MODEL_DIR, 'hybrid_classifier.pkl'))
        joblib.dump(catboost_regressor, os.path.join(MODEL_DIR, 'hybrid_regressor.pkl'))
        
        models['Hybrid CatBoost'] = {
            'classifier': catboost_classifier,
            'regressor': catboost_regressor
        }
        
        results['Hybrid CatBoost'] = {
            'classification': class_metrics,
            'regression': reg_metrics
        }
        
        return models, results
        
    except Exception as e:
        logging.error(f"Error training Hybrid models: {str(e)}")
        raise

def train_models(df, selected_models):
    """Train selected models and return results"""
    try:
        # Preprocess data
        df_processed, label_encoders = load_and_preprocess_data(df)
        
        # Prepare features and targets
        target_cols = ['Network Type', 'Signal Strength (dBm)']
        feature_cols = [col for col in df_processed.columns if col not in target_cols]
        
        X = df_processed[feature_cols]
        y_class = df_processed['Network Type']
        y_reg = df_processed['Signal Strength (dBm)']
        
        # Train-test split
        X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
            X, y_class, y_reg, test_size=0.2, random_state=42, stratify=y_class
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Save scaler and label encoders
        joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
        joblib.dump(label_encoders, os.path.join(MODEL_DIR, 'label_encoders.pkl'))
        
        all_models = {}
        all_classification_results = {}
        all_regression_results = {}
        
        # Train selected models
        for model_name in selected_models:
            models = {}
            results = {}
            
            if model_name == 'Ridge':
                models, results = train_ridge_models(
                    X_train_scaled, X_test_scaled, y_class_train, y_class_test, y_reg_train, y_reg_test
                )
            elif model_name == 'Decision Tree':
                models, results = train_decision_tree_models(
                    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test
                )
            elif model_name == 'Hybrid CatBoost':
                models, results = train_hybrid_mlp_catboost(
                    X_train_scaled, X_test_scaled, y_class_train, y_class_test, y_reg_train, y_reg_test
                )
            
            all_models.update(models)
            for model_key, model_results in results.items():
                all_classification_results[model_key] = model_results['classification']
                all_regression_results[model_key] = model_results['regression']
        
        return all_models, all_classification_results, all_regression_results
        
    except Exception as e:
        logging.error(f"Error in train_models: {str(e)}")
        raise

def predict_new_data(input_data, model_name, models):
    """Make predictions on new data"""
    try:
        if model_name not in models:
            raise ValueError(f"Model {model_name} not found")
        
        # Load preprocessors
        scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
        label_encoders = joblib.load(os.path.join(MODEL_DIR, 'label_encoders.pkl'))
        
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Apply same preprocessing
        for col in input_df.select_dtypes(include='object').columns:
            if col in label_encoders and col != 'Network Type':
                le = label_encoders[col]
                input_df[col] = le.transform(input_df[col].astype(str))
        
        # Fill missing values with 0 (or you could use training means)
        input_df = input_df.fillna(0)
        
        # Scale features if model requires it
        if model_name in ['Ridge', 'Hybrid CatBoost']:
            input_scaled = scaler.transform(input_df)
            input_processed = input_scaled
        else:
            input_processed = input_df.values
        
        # Get models
        model_set = models[model_name]
        classifier = model_set['classifier']
        regressor = model_set['regressor']
        
        # Make predictions
        class_pred = classifier.predict(input_processed)[0]
        reg_pred = regressor.predict(input_processed)[0]
        
        # Decode class prediction
        if 'Network Type' in label_encoders:
            class_pred_decoded = label_encoders['Network Type'].inverse_transform([class_pred])[0]
        else:
            class_pred_decoded = class_pred
        
        return {
            'network_type': class_pred_decoded,
            'signal_strength': float(reg_pred),
            'model_used': model_name
        }
        
    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        raise

def evaluate_models(models, X_test, y_class_test, y_reg_test):
    """Evaluate trained models and return metrics"""
    try:
        results = {}
        
        for model_name, model_set in models.items():
            classifier = model_set['classifier']
            regressor = model_set['regressor']
            
            # Make predictions
            y_class_pred = classifier.predict(X_test)
            y_reg_pred = regressor.predict(X_test)
            
            # Classification metrics
            class_metrics = {
                'accuracy': accuracy_score(y_class_test, y_class_pred) * 100,
                'precision': precision_score(y_class_test, y_class_pred, average='macro') * 100,
                'recall': recall_score(y_class_test, y_class_pred, average='macro') * 100,
                'f1_score': f1_score(y_class_test, y_class_pred, average='macro') * 100
            }
            
            # Regression metrics
            reg_metrics = {
                'mae': mean_absolute_error(y_reg_test, y_reg_pred),
                'mse': mean_squared_error(y_reg_test, y_reg_pred),
                'rmse': np.sqrt(mean_squared_error(y_reg_test, y_reg_pred)),
                'r2': r2_score(y_reg_test, y_reg_pred)
            }
            
            results[model_name] = {
                'classification': class_metrics,
                'regression': reg_metrics
            }
        
        return results
        
    except Exception as e:
        logging.error(f"Error evaluating models: {str(e)}")
        raise
