import json
import os
import logging
import xgboost as xgb
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def model_fn(model_dir):
    """Load the XGBoost model from the model directory."""
    try:
        logger.info(f"Loading model from {model_dir}")
        
        # List all files in model directory for debugging
        all_files = os.listdir(model_dir)
        logger.info(f"Files in model directory: {all_files}")
        
        model_path = os.path.join(model_dir, 'xgboost_model.json')
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        # Load XGBoost model
        model = xgb.Booster()
        model.load_model(model_path)
        logger.info("Successfully loaded XGBoost model")
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def input_fn(request_body, request_content_type='application/json'):
    """Parse input data for inference."""
    try:
        logger.info(f"Processing input with content type: {request_content_type}")
        
        if request_content_type != 'application/json':
            raise ValueError(f"Unsupported content type: {request_content_type}")
        
        # Parse JSON input
        if isinstance(request_body, str):
            input_data = json.loads(request_body)
        else:
            input_data = request_body
            
        logger.info(f"Parsed input data: {input_data}")
        
        # Handle different input formats
        if isinstance(input_data, dict):
            # Single prediction dictionary - convert to list
            input_data = [input_data]
        elif not isinstance(input_data, list):
            raise ValueError(f"Input must be a dictionary or list. Got: {type(input_data)}")
        
        # Expected features in exact order from training
        expected_features = [
            'esent', 'eopenrate', 'eclickrate', 'avgorder', 'ordfreq', 
            'paperless', 'refill', 'doorstep', 'created_year', 
            'created_month', 'created_day', 'account_age_days'
        ]
        
        features_list = []
        for item in input_data:
            if not isinstance(item, dict):
                raise ValueError(f"Each input item must be a dictionary. Got: {type(item)}")
            
            # Extract features in exact order
            features = []
            for feature_name in expected_features:
                value = item.get(feature_name, 0.0)  # Default to 0.0 if missing
                try:
                    features.append(float(value))
                except (ValueError, TypeError):
                    logger.warning(f"Could not convert {feature_name}={value} to float, using 0.0")
                    features.append(0.0)
            
            features_list.append(features)
            logger.info(f"Extracted features: {features}")
        
        # Convert to numpy array and create DMatrix
        features_array = np.array(features_list, dtype=np.float32)
        logger.info(f"Created features array with shape: {features_array.shape}")
        
        dmatrix = xgb.DMatrix(features_array)
        logger.info(f"Created DMatrix successfully")
        
        return dmatrix
        
    except Exception as e:
        logger.error(f"Error in input_fn: {str(e)}")
        raise

def predict_fn(input_data, model):
    """Make prediction using the loaded model."""
    try:
        logger.info("Making prediction...")
        
        # Make prediction
        predictions = model.predict(input_data)
        logger.info(f"Predictions: {predictions}")
        
        return predictions
        
    except Exception as e:
        logger.error(f"Error in predict_fn: {str(e)}")
        raise

def output_fn(predictions, accept='application/json'):
    """Format the prediction output."""
    try:
        logger.info(f"Formatting output with accept type: {accept}")
        
        # Convert predictions to proper format
        if isinstance(predictions, np.ndarray):
            pred_list = predictions.tolist()
        else:
            pred_list = [float(predictions)]
        
        # Create simple output format
        result = {
            'predictions': pred_list
        }
        
        logger.info(f"Final output: {result}")
        return json.dumps(result)
        
    except Exception as e:
        logger.error(f"Error in output_fn: {str(e)}")
        raise
