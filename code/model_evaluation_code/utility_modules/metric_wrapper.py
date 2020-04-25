# This module is used to wrap a model and its metrics into a single object
# for comparison of models and finding the best ones

class metric_wrapper:
    def __init__(self, rmse, mae, accuracy, model = None):
        self.rmse = rmse
        self.mae = mae
        self.accuracy = accuracy
        self.model = model
        
    def copy(self, rmse, mae, accuracy, model = None):
        self.rmse = rmse
        self.mae = mae
        self.accuracy = accuracy
        self.model = model
        
    def print_values(self):
        print('mae: ', self.mae)
        print('rmse: ', self.rmse)
        print('accuracy: ', self.accuracy)
        
    def get_model(self):
        return self.model
