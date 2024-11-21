import tensorflow as tf
import numpy as np
from .metrics import rmse, pcc

class PredictionCallback(tf.keras.callbacks.Callback):

    def __init__(self, validation_data, interval=1):


        super().__init__()
        self.validation_data = validation_data
        self.interval = interval
        self.predictions = []
        
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.interval == 0:
            x_val, y_val = self.validation_data
            y_pred = self.model.predict(x_val)
            

            rmse_val = rmse(y_val, y_pred)
            pcc_val = pcc(y_val, y_pred)
            

            self.predictions.append({
                'epoch': epoch + 1,
                'rmse': rmse_val,
                'pcc': pcc_val
            })
            
            print(f'\nEpoch {epoch + 1}:')
            print(f'RMSE: {rmse_val:.4f}')
            print(f'PCC: {pcc_val:.4f}')
