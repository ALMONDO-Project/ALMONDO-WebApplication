import numpy as np
import pandas as pd
from io import StringIO
from typing import List, Optional
from exceptions.custom_exceptions import ConfigurationError, ValidationError, FileUploadError
from flask import current_app
import logging

logger = logging.getLogger(__name__)

def get_int(form_data, key, min_val: Optional[int] = None) -> int:
    if key not in form_data:
        raise ValidationError(f"Missing parameter in form data for initial_status_generator() '{key}'", field=key)
    try:
        value = int(form_data[key])
    except Exception:
        raise ValidationError(f"Parameter '{key}' in form data for initial_status_generator() must be an integer.", field=key)
    if min_val is not None and value < min_val:
        raise ValidationError(f"Parameter '{key}' in form data for initial_status_generator() must be >= {min_val}.", field=key)
    return value


def get_float(form_data, key, min_val: Optional[float] = None, max_val: Optional[float] = None) -> float:
    if key not in form_data:
        raise ValidationError(f"Missing parameter '{key}' in form data for initial_status_generator()", field=key)
    try:
        value = float(form_data[key])
    except Exception:
        raise ValidationError(f"Parameter '{key}' in form data for initial_status_generator() must be a float.", field=key)
    if min_val is not None and value < min_val:
        raise ValidationError(f"Parameter '{key}' in form data for initial_status_generator() must be >= {min_val}.", field=key)
    if max_val is not None and value > max_val:
        raise ValidationError(f"Parameter '{key}' in form data for initial_status_generator() must be <= {max_val}.", field=key)
    return value

def generate_initial_status(initial_status_type: str, form_data: dict, 
                            files: Optional[dict], num_nodes: int, 
                            validate_range: bool = True) -> List[float]:
    """
    Generate initial status for network nodes based on the specified type and parameters.
    
    Args:
        initial_status_type (str): Type of initial status ('uniform', 'unbiased', 'gaussian_mixture', 'user_defined')
        form_data (dict): Form data containing parameters
        files (dict): Files from the request (for file-based status)
        num_nodes (int): Number of nodes in the graph
        validate_range (bool): Whether to validate that all values are between 0 and 1
    
    Returns:
        initial_status (list): Initial status for the network nodes
    """
    if not isinstance(initial_status_type, str) or not initial_status_type:
            raise ValidationError("Invalid or missing initial_status_type.", field='initial_status_type')

    if not isinstance(num_nodes, int) or num_nodes <= 0:
        raise ValidationError("Number of nodes in the graph must be a positive integer for initial_status_generator().", field='num_nodes')

    try:
       
        if initial_status_type == "uniform":
            try: 
                min_range = get_float(form_data, 'minRangeUniformDistribution')
                max_range = get_float(form_data, 'maxRangeUniformDistribution')
            except Exception:
                raise ValidationError("Uniform distribution  in initial_status_generator() requires 'minRangeUniformDistribution' and 'maxRangeUniformDistribution'.")
            if min_range >= max_range:
                raise ValidationError("In initial_status_generator() uniform distribution min must be < max.", field="uniform_range")
            # Random uniform initial status for each node in the interval
            weights = np.random.uniform(low=min_range, high=max_range, size=num_nodes)
            # initial_status = weights.tolist()
            
        elif initial_status_type == "unbiased":
            try:
                unbiased_value = get_float(form_data,'unbiasedValue')
            except Exception:
                raise ValidationError("Unbiased distribution in initial_status_generator() requires 'unbiasedValue'.", field="unbiasedValue")
            # Unbiased initial status for each node
            weights = np.full(num_nodes, unbiased_value)
            # initial_status = weights.tolist()
            
        elif initial_status_type == "gaussian_mixture":
            file = files.get('status')
            
            if not file or file.filename == '':
                raise FileUploadError("File not valid or not selected for gaussian_mixture in files form for initial_status_generator().")

            current_app.logger.info(f"Received file for gaussian_mixture: {file}")
            
            try:
                content = file.read().decode('utf-8')
                gaussian_params = {}
                
                for row in content.splitlines():
                    row = row.strip()
                    if not row:
                        continue
                        
                    key, vals = row.split(':', 1)
                    values_list = [float(x.strip()) for x in vals.split(',')]
                    gaussian_params[key.strip()] = values_list
                
                # Security control on keys and values
                required_keys = ['means', 'stds', 'weights']
                if not isinstance(gaussian_params, dict) or not all(key in gaussian_params for key in required_keys):
                    missing = [key for key in required_keys if key not in gaussian_params]
                    raise ValidationError(f"In initial_status_generator() missing keys in Gaussian mixture file: {missing}")
                    
                
                # Extract parameters
                means = gaussian_params['means']
                stds = gaussian_params['stds']
                ws = gaussian_params['weights']
                
                # Validation
                if len(means) != len(stds) or len(means) != len(ws):
                    raise ValidationError("In initial_status_generator() in gaussian mixture means, stds, and weights must have the same length")
                if not np.isclose(sum(ws), 1.0):
                    raise ValidationError("In initial_status_generator() in gaussian mixture the sum of weights must equal 1.")

                # Generate Gaussian mixture
                gaussians = [np.random.normal(loc=means[i], scale=stds[i], size=num_nodes) for i in range(len(means))]
                mixture = np.zeros(num_nodes)
                
                # Sample according to the weights
                for i, w in enumerate(ws):
                    sample_size = int(num_nodes * w)
                    if sample_size > 0:
                        indices = np.random.choice(num_nodes, size=sample_size, replace=False)
                        mixture[indices] = gaussians[i][indices]
                
                # Ensure values are between 0 and 1
                weights = np.clip(mixture, 0, 1)
                # initial_status = weights.tolist()
                
            except ValueError as ve:
                raise ValidationError(f"Invalid Gaussian mixture file format in initial_status_generator(): {ve}", field="status_file")
            except Exception as e:
                raise FileUploadError(f"Error while reading Gaussian mixture file in initial_status_generator(): {e}")

        elif initial_status_type == "user_defined":
            file = files.get('status')
            
            if not file or file.filename == "":
                raise FileUploadError("No file provided for user_defined initial status.")
            
            try: 
                if not file.filename.lower().endswith(('.csv', '.txt')):
                    raise ValidationError("Unsupported file format for user defined initial status. Use .csv or .txt.", field="status_file")
                
                current_app.logger.info(f"Received file for user_defined initial status: {file}")

                content = file.read().decode('utf-8')
                
                if file.filename.lower().endswith(".csv"):
                    df = pd.read_csv(StringIO(content))
                    # Ensure there is only one column
                    if df.shape[1] != 1:
                        raise ValidationError("In user defined initial status generation CSV must have exactly one column of numbers.")
                    # Ensure we're working with numerical data
                    column = df.columns[0]
                    if not np.issubdtype(df[column].dtype, np.number):
                        raise ValidationError("In user defined initial status generation CSV column must contain numeric data.")

                    # Ensure no NaN values
                    weights = df[column].dropna().values # df.iloc[:, 0].dropna().values
                    # initial_status = weights.tolist()
                    
                else: # TXT if file.filename.lower().endswith(".txt"):
                    initial_status = [
                        float(line.strip()) 
                        for line in content.splitlines() 
                        if line.strip()
                    ]

            except ValueError as ve:
                raise ValidationError(f"Invalid numeric value in user_defined file for initial status: {ve}", field="status_file")
                    
            except Exception as e:
                raise FileUploadError(f"Error while reading user_defined file for initial status: {e}")
    
        else:
            raise ValidationError(f"Unsupported initial_status_type: {initial_status_type}", field="initial_status_type")

        if initial_status_type != 'user_defined':
            initial_status = weights.tolist()
            
        # Add validation at the end if requested
        if validate_range and not all(0 <= val <= 1 for val in initial_status):
                raise ValidationError("All initial status values must be between 0 and 1.", field="values")
        
        if len(initial_status) != num_nodes:
            raise ValidationError(f"Initial status length ({len(initial_status)}) does not match num_nodes ({num_nodes}).")
        
        return initial_status
        
    except ValidationError:
        raise
    except FileUploadError:
        raise
    except Exception as e:
        raise ConfigurationError(f"Unexpected error generating initial status: {e}")
