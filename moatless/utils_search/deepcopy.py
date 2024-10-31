from collections.abc import Mapping, Sequence
import types
from pydantic import BaseModel
import copy
from types import MethodType, FunctionType


def clone_state(state):
    def _clone_recursive(obj, memo):
        obj_id = id(obj)
        if obj_id in memo:
            return memo[obj_id]

        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj

        if isinstance(obj, (list, tuple)):
            new_obj = []
            memo[obj_id] = new_obj
            new_obj.extend(_clone_recursive(item, memo) for item in obj)
            return type(obj)(new_obj)  # Cast back to original type (list or tuple)

        if isinstance(obj, dict):
            new_obj = {}
            memo[obj_id] = new_obj
            for key, value in obj.items():
                new_key = _clone_recursive(key, memo)
                new_value = _clone_recursive(value, memo)
                new_obj[new_key] = new_value
            return new_obj

        if isinstance(obj, set):
            new_obj = set()
            memo[obj_id] = new_obj
            new_obj.update(_clone_recursive(item, memo) for item in obj)
            return new_obj

        # For all other objects, use copy.deepcopy
        try:
            new_obj = copy.deepcopy(obj, memo)
            memo[obj_id] = new_obj
            print(f"Cloned: {obj}")
            return new_obj
        except RecursionError:
            print(f"RecursionError: {obj}")
            # If we still hit a RecursionError, fall back to shallow copy
            return copy.copy(obj)

    return _clone_recursive(state, {})


def safe_deepcopy(obj, memo=None, _depth=0, _max_depth=1000):
    if memo is None:
        memo = {}

    if _depth > _max_depth:
        raise RuntimeError(f"Maximum recursion depth exceeded ({_max_depth})")

    obj_id = id(obj)
    if obj_id in memo:
        return memo[obj_id]

    # Handle Pydantic models
    if isinstance(obj, BaseModel):
        new_obj = obj.model_copy(deep=True)
        memo[obj_id] = new_obj
        return new_obj

    try:
        # Try to use object's __deepcopy__ method if available
        if hasattr(obj, '__deepcopy__'):
            return obj.__deepcopy__(memo)
    except Exception:
        pass  # Fall back to our custom logic if __deepcopy__ fails

    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj

    if isinstance(obj, type):
        # For non-Pydantic classes, return the class as-is
        return obj

    if isinstance(obj, Mapping):
        new_obj = obj.__class__()
        memo[obj_id] = new_obj
        for key, value in obj.items():
            new_obj[safe_deepcopy(key, memo, _depth + 1)] = safe_deepcopy(value, memo, _depth + 1)
        return new_obj

    if isinstance(obj, tuple):
        # Handle tuples separately
        new_obj = tuple(safe_deepcopy(item, memo, _depth + 1) for item in obj)
        memo[obj_id] = new_obj
        return new_obj

    if isinstance(obj, Sequence) and not isinstance(obj, str):
        new_obj = obj.__class__()
        memo[obj_id] = new_obj
        if hasattr(new_obj, 'extend'):
            new_obj.extend(safe_deepcopy(item, memo, _depth + 1) for item in obj)
        else:
            # Fallback for sequences without extend method
            for item in obj:
                new_obj.append(safe_deepcopy(item, memo, _depth + 1))
        return new_obj

    if hasattr(obj, '__dict__'):
        cls = obj.__class__
        try:
            new_obj = cls.__new__(cls)
        except TypeError:
            # If __new__ doesn't work, return the object as-is
            return obj

        memo[obj_id] = new_obj
        for key, value in obj.__dict__.items():
            if not isinstance(value, (types.FunctionType, types.MethodType, classmethod, staticmethod, property)):
                setattr(new_obj, key, safe_deepcopy(value, memo, _depth + 1))
        
        # Handle __slots__ if present
        if hasattr(cls, '__slots__'):
            for slot in cls.__slots__:
                if hasattr(obj, slot):
                    setattr(new_obj, slot, safe_deepcopy(getattr(obj, slot), memo, _depth + 1))
        
        return new_obj

    # If we can't handle it, try the standard deepcopy
    try:
        return copy.deepcopy(obj, memo)
    except Exception as e:
        # If all else fails, return the object as-is
        return obj


def custom_deepcopy(obj, memo=None):
    if memo is None:
        memo = {}

    obj_id = id(obj)
    if obj_id in memo:
        return memo[obj_id]

    # Handle Pydantic models and classes with __deepcopy__
    if isinstance(obj, BaseModel) or hasattr(obj, '__deepcopy__'):
        try:
            new_obj = obj.__deepcopy__(memo)
            memo[obj_id] = new_obj
            return new_obj
        except TypeError as e:
            if str(e).endswith("missing 1 required positional argument: 'memo'"):
                # If the __deepcopy__ method doesn't accept memo, call it without memo
                new_obj = obj.__deepcopy__()
                memo[obj_id] = new_obj
                return new_obj
            else:
                print(f"Error deep copying object: {e}")
                return obj

    # Handle basic types
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj

    # Handle methods and functions
    if isinstance(obj, (MethodType, FunctionType)):
        return obj

    # Handle lists, tuples, and sets
    if isinstance(obj, (list, tuple, set)):
        new_obj = obj.__class__(custom_deepcopy(item, memo) for item in obj)
        memo[obj_id] = new_obj
        return new_obj

    # Handle dictionaries and dict-like objects
    if isinstance(obj, dict) or hasattr(obj, 'items'):
        new_obj = obj.__class__()
        memo[obj_id] = new_obj
        for k, v in obj.items():
            new_obj[custom_deepcopy(k, memo)] = custom_deepcopy(v, memo)
        return new_obj

    # Handle objects with a custom __deepcopy__ method
    if hasattr(obj, '__deepcopy__'):
        return obj.__deepcopy__(memo)

    # Handle special cases (add more as needed)
    if obj.__class__.__name__ in ['CodeIndex', 'Transitions']:
        print(f"Special case: {obj.__class__.__name__}")
        new_obj = copy.copy(obj)  # Shallow copy for special cases
        memo[obj_id] = new_obj
        return new_obj

    # For all other objects, create a new instance and copy attributes
    try:
        new_obj = obj.__class__.__new__(obj.__class__)
        memo[obj_id] = new_obj
        
        if hasattr(obj, '__dict__'):
            for key, value in obj.__dict__.items():
                setattr(new_obj, key, custom_deepcopy(value, memo))
        elif hasattr(obj, '__slots__'):
            for slot in obj.__slots__:
                if hasattr(obj, slot):
                    setattr(new_obj, slot, custom_deepcopy(getattr(obj, slot), memo))
        
        return new_obj
    except Exception as e:
        print(f"Failed to deepcopy {obj.__class__.__name__}: {str(e)}")
        return copy.copy(obj)  # Fallback to shallow copy if deepcopy fails
    
