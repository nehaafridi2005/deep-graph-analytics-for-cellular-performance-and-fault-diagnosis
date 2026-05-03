import os
import logging
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
import pandas as pd
import numpy as np
import joblib
from ml_utils import (
    load_and_preprocess_data, train_models, evaluate_models, 
    create_eda_plots, predict_new_data, MODEL_DIR, RESULTS_DIR
)
import json
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "fallback_secret_key_for_development")

# Create necessary directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs('static/plots', exist_ok=True)

# Global variables to store data and results
app_data = {
    'df': None,
    'test_df': None,
    'models': {},
    'classification_results': {},
    'regression_results': {},
    'eda_plots': {}
}

def load_data():
    """Load the datasets"""
    try:
        app_data['df'] = pd.read_csv('data/signal_metrics_1755883814700.csv')
        app_data['test_df'] = pd.read_csv('data/TestData_1755883814701.csv')
        app.logger.info("Data loaded successfully")
        return True
    except Exception as e:
        app.logger.error(f"Error loading data: {str(e)}")
        return False

@app.route('/')
def home():
    """Home page"""
    if app_data['df'] is None:
        if not load_data():
            flash('Error loading data files. Please ensure CSV files are in the data directory.', 'error')
            return render_template('home.html', data_loaded=False)
    
    data_info = {
        'total_records': len(app_data['df']),
        'features': list(app_data['df'].columns),
        'network_types': app_data['df']['Network Type'].unique().tolist() if 'Network Type' in app_data['df'].columns else [],
        'test_records': len(app_data['test_df']) if app_data['test_df'] is not None else 0
    }
    
    return render_template('home.html', data_loaded=True, data_info=data_info)

@app.route('/eda')
def eda():
    """Exploratory Data Analysis page"""
    if app_data['df'] is None:
        flash('Data not loaded. Please go to home page first.', 'error')
        return redirect(url_for('home'))
    
    try:
        # Generate EDA plots
        eda_results = create_eda_plots(app_data['df'])
        app_data['eda_plots'] = eda_results
        
        return render_template('eda.html', eda_results=eda_results)
    except Exception as e:
        app.logger.error(f"Error in EDA: {str(e)}")
        flash(f'Error generating EDA plots: {str(e)}', 'error')
        return render_template('eda.html', eda_results={})

@app.route('/classification')
def classification():
    """Classification performance page"""
    if app_data['df'] is None:
        flash('Data not loaded. Please go to home page first.', 'error')
        return redirect(url_for('home'))
    
    return render_template('classification.html', results=app_data['classification_results'])

@app.route('/regression')
def regression():
    """Regression performance page"""
    if app_data['df'] is None:
        flash('Data not loaded. Please go to home page first.', 'error')
        return redirect(url_for('home'))
    
    return render_template('regression.html', results=app_data['regression_results'])

@app.route('/train_models', methods=['POST'])
def train_models_route():
    """Train selected models"""
    if app_data['df'] is None:
        return jsonify({'error': 'Data not loaded'}), 400
    
    try:
        json_data = request.get_json() or {}
        selected_models = json_data.get('models', [])
        app.logger.info(f"Training models: {selected_models}")
        
        # Train models and get results
        models, classification_results, regression_results = train_models(
            app_data['df'], selected_models
        )
        
        # Store results
        app_data['models'].update(models)
        app_data['classification_results'] = classification_results
        app_data['regression_results'] = regression_results
        
        return jsonify({
            'success': True,
            'message': f'Successfully trained {len(selected_models)} models',
            'classification_results': classification_results,
            'regression_results': regression_results
        })
    
    except Exception as e:
        app.logger.error(f"Error training models: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/prediction')
def prediction():
    """Prediction page"""
    if app_data['df'] is None:
        flash('Data not loaded. Please go to home page first.', 'error')
        return redirect(url_for('home'))
    
    # Check if models are trained
    if not app_data['models']:
        flash('No models trained yet. Please train models first.', 'warning')
    
    # Get feature names for the form
    if app_data['test_df'] is not None:
        sample_data = app_data['test_df'].iloc[0].to_dict()
        # Remove target columns if present
        sample_data.pop('Network Type', None)
        sample_data.pop('Signal Strength (dBm)', None)
    else:
        sample_data = {}
    
    return render_template('prediction.html', 
                         models=list(app_data['models'].keys()),
                         sample_data=sample_data)

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions"""
    try:
        # Get form data
        model_name = request.form.get('model')
        input_data = {}
        
        # Extract input features
        for key, value in request.form.items():
            if key != 'model' and value:
                try:
                    input_data[key] = float(value)
                except ValueError:
                    input_data[key] = value
        
        if not input_data:
            flash('Please provide input data for prediction.', 'error')
            return redirect(url_for('prediction'))
        
        if model_name not in app_data['models']:
            flash(f'Model {model_name} not found. Please train the model first.', 'error')
            return redirect(url_for('prediction'))
        
        # Make prediction
        predictions = predict_new_data(input_data, model_name, app_data['models'])
        
        flash(f'Prediction successful!', 'success')
        return render_template('prediction.html',
                             models=list(app_data['models'].keys()),
                             sample_data=input_data,
                             predictions=predictions,
                             selected_model=model_name)
    
    except Exception as e:
        app.logger.error(f"Error making prediction: {str(e)}")
        flash(f'Error making prediction: {str(e)}', 'error')
        return redirect(url_for('prediction'))

@app.route('/load_test_sample/<int:index>')
def load_test_sample(index):
    """Load a sample from test data"""
    if app_data['test_df'] is None or index >= len(app_data['test_df']):
        return jsonify({'error': 'Invalid sample index'}), 400
    
    sample = app_data['test_df'].iloc[index].to_dict()
    # Remove target columns if present
    sample.pop('Network Type', None)
    sample.pop('Signal Strength (dBm)', None)
    
    return jsonify(sample)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
