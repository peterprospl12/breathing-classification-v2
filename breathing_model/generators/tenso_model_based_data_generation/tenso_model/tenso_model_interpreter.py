import tensorflow as tf
import numpy as np

class TFLiteModelWrapper:
    def __init__(self, model_path):
        """
        Initialize the TFLite model wrapper.
        
        :param model_path: Path to the .tflite model file.
        """
        self.model_path = model_path
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, input_data):
        if self.interpreter is None:
            raise ValueError("The TFLite interpreter is not initialized.")
        
        if len(input_data.shape) == 2:
            input_data = np.expand_dims(input_data, axis=1)


        predictions = []
        for sample in input_data:
            # Set input tensor
            sample = np.expand_dims(sample, axis=0).astype(self.input_details[0]['dtype'])

            self.interpreter.set_tensor(self.input_details[0]['index'], sample)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]['index'])
            predictions.append(np.argmax(output))
        return np.array(predictions)

