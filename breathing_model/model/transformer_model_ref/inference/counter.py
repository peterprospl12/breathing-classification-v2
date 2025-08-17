from breathing_model.model.transformer_model_ref.utils import BreathType


class BreathCounter:
    def __init__(self):
        self.inhale = 0
        self.exhale = 0
        self.last_predict = None

    def update(self, prediction: int):
        if self.last_predict == prediction:
            return
        if prediction == BreathType.EXHALE:
            self.exhale += 1
        elif prediction == BreathType.INHALE:
            self.inhale += 1

        self.last_predict = prediction

    def reset(self):
        self.inhale = 0
        self.exhale = 0
        self.last_predict = None

    def __str__(self) -> str:
        return f"Inhales: {self.inhale}, Exhales: {self.exhale}"
