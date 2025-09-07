from breathing_model.model.exhale_only_detection.utils import BreathType


class BreathCounter:
    def __init__(self):
        self.exhale = 0
        self.last_predict = None

    def update(self, prediction: int):
        if self.last_predict == prediction:
            return
        if prediction == BreathType.EXHALE:
            self.exhale += 1
        self.last_predict = prediction

    def reset(self):
        self.exhale = 0
        self.last_predict = None

    def __str__(self) -> str:
        return f"Exhales: {self.exhale}"