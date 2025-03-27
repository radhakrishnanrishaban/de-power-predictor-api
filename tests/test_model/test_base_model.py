import pytest
import pandas as pd
from src.model.base import BaseModel

class TestModel(BaseModel):
    """Test implementation of BaseModel"""
    def train(self, X, y):
        pass
    
    def predict(self, X):
        pass
    
    def save(self, path):
        pass
    
    def load(self, path):
        pass
    
    def get_params(self):
        return {}

def test_base_model_abstract():
    """Test that BaseModel cannot be instantiated directly"""
    with pytest.raises(TypeError):
        BaseModel()

def test_base_model_implementation():
    """Test that BaseModel can be implemented"""
    model = TestModel()
    assert isinstance(model, BaseModel)