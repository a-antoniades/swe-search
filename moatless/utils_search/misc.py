import logging
import os
import sys
import json
import tiktoken
from datetime import datetime
import logging


logger = logging.getLogger(__name__)

def update_instance_id(predictions_path, current_instance_id, new_prediction):
    # Read existing predictions
    with open(predictions_path, 'r') as file:
        predictions = [json.loads(line) for line in file]
    
    # Find and update the prediction for the current instance_id
    updated = False
    for i, pred in enumerate(predictions):
        if pred['instance_id'] == current_instance_id:
            predictions[i] = new_prediction
            updated = True
            break
    
    # If the instance_id wasn't found, append the new prediction
    if not updated:
        predictions.append(new_prediction)
    
    # Write all predictions back to the file
    with open(predictions_path, 'w') as file:
        for pred in predictions:
            json.dump(pred, file)
            file.write('\n')

def save_json_dict(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def save_to_json(files, filename):
    # Get the absolute path of the file
    full_path = os.path.abspath(filename)
    
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    try:
        with open(full_path, 'w') as f:
            try:
                import json5
                json5.dump(files, f, indent=2, quote_keys=True, trailing_commas=False)
            except ImportError:
                json.dump(files, f, indent=2, ensure_ascii=False)

        logger.info(f"LLM output saved to {full_path}")
    except Exception as e:
        logger.error(f"Error saving interactions to JSON: {e}")
        logger.error(f"Attempted to save to: {full_path}")
        
def deep_get(data, keys, default=None):
    """
    Access nested dictionary items with a delimited string of keys.
    
    :param data: The dictionary to search
    :param keys: A string of keys separated by a delimiter
    :param default: The default value to return if the key is not found
    :return: The value if found, else the default
    """
    keys = keys.split('.')
    for key in keys:
        if isinstance(data, dict):
            data = data.get(key, default)
        else:
            return default
    return data

def safe_repr(obj, max_depth=10):
    seen = set()

    def _repr(obj, depth):
        if depth > max_depth:
            return "..."
        
        obj_id = id(obj)
        if obj_id in seen:
            return "..."
        seen.add(obj_id)

        if isinstance(obj, dict):
            items = []
            for k, v in obj.items():
                items.append(f"{_repr(k, depth + 1)}: {_repr(v, depth + 1)}")
            return "{" + ", ".join(items) + "}"
        elif isinstance(obj, (list, tuple, set)):
            items = [_repr(item, depth + 1) for item in obj]
            return f"{type(obj).__name__}({', '.join(items)})"
        elif hasattr(obj, '__dict__'):
            items = []
            for k, v in obj.__dict__.items():
                items.append(f"{k}={_repr(v, depth + 1)}")
            return f"{type(obj).__name__}({', '.join(items)})"
        else:
            return repr(obj)

    return _repr(obj, 0)

def configure_logger(log_to_file=True, log_file='mcts_search.log', level=logging.INFO):
    logger = logging.getLogger('mcts')
    logger.setLevel(level)

    logging.getLogger('mcts.reward').setLevel(logging.WARNING)

    logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if log_to_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger
