import numpy as np
import pandas as pd
import tensorflow as tf
import csv
import os

WINNDOW_SIZE = 6
ROW_DATA_PATH = "../data/tenso_row_converted_data"
ROW_CONVERTED_DATA_PATH = "../data/tenso_row_converted_data"
LABELD_DATA_PATH = "../data/tenso_labeled_data"
SAVED_MODEL_PATH = "./saved_models/GRUModel_tens.tflite"

class TFLiteModel:
    def __init__(self, model_path = SAVED_MODEL_PATH):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

    def predict(self, file_name):
        self.convert_timestamps(file_name=file_name)
        data_path = os.path.join(ROW_CONVERTED_DATA_PATH, file_name)
        times, numbers = self.load_data(data_path)
        
        numbers = self.moving_average(numbers, WINNDOW_SIZE)
        number = self.normalize(numbers, 150)

        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        if len(X.shape) == len(input_details[0]['shape']):
            X = np.expand_dims(X, axis=0)
        
        X = X.astype(input_details[0]['dtype'])

        self.interpreter.set_tensor(input_details[0]['index'], X)

        self.interpreter.invoke()

        output = self.interpreter.get_tensor(output_details[0]['index'])

        return output
    
    def load_data(self, filename):
        numbers = []
        times = []
        with open(filename) as file:
            data = list(csv.reader(file))[1:]
            for data_line in data:
                numbers.append(float(data_line[1]))
                times.append(float(data_line[0]))
        return times, numbers
    
    def save_tagged_data(self, data, tags, time, file_name):
        output_path = os.path.join(LABELD_DATA_PATH, file_name)
        with open(output_path, "w") as file:
            writer = csv.writer(file)
            writer.writerow(["time", "data-raw", "tag"])
            for i in range(len(data)):
                writer.writerow([time[i], data[i], tags[i]])
        print(f"Plik zapisany w: {output_path}")

    def convert_timestamps(self, file_name):
        """
        Convert timestamps in a given file to seconds since the start and save the output as a CSV file
        in a specified output directory.

        Args:
        - file_name (str): Name of the input file (without directory path).
        """
        input_path = os.path.join(ROW_DATA_PATH, file_name)
        output_path = os.path.join(ROW_CONVERTED_DATA_PATH, file_name)

        df = pd.read_csv(input_path, delimiter=",", header=None)
        
        df[0] = pd.to_datetime(df[0])
        
        df[0] = (df[0] - df[0].min()).dt.total_seconds()

        df.columns = ["seconds", "data-raw"]
        
        df.to_csv(output_path, index=False)

        print(f"Plik zapisany w: {output_path}")

        

    def normalize(self, numbers, normalization_range):
        
        def normalize_window(window):
            min_val = min(window)
            max_val = max(window)
            range_val = max_val - min_val
            if range_val == 0:
                return [0 for _ in window]

            normalized = [(-1 + 2 * (x - min_val) / range_val) for x in window]
            return normalized
        
        normalized_values = []

        for i in range(len(numbers)):
            window = numbers[max(0, i - normalization_range) : i]
            try:
                normalized_window_values = normalize_window(window)
                normalized_values.append(normalized_window_values[-1])
            except ValueError:
                continue

        return normalized_values


    def moving_average(self, data, window_size):
        return np.convolve(data, np.ones(window_size) / window_size, mode="valid")
