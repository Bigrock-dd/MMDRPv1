import tensorflow as tf

class BaseModel:
    
    def __init__(self):
        self.model = None
        
    def build(self):
        raise NotImplementedError
        
    def save(self, filepath):
        if self.model:
            self.model.save(filepath)
            
    def load(self, filepath):
        self.model = tf.keras.models.load_model(filepath)
        
    def predict(self, x):
        if not self.model:
            raise ValueError("Model not built yet!")
        return self.model.predict(x)