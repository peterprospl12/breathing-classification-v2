class BreathCounter:
    def __init__(self):
        self.inhale = 0
        self.exhale = 0
        self.last_predict = None

    def update(self, prediction):
        if self.last_predict != prediction:
            if prediction == 0:
                self.exhale += 1
            elif prediction == 1:
                self.inhale += 1
        self.last_predict = prediction

    def reset(self):
        self.inhale = 0
        self.exhale = 0

    def __str__(self):
        return f"Inhales: {self.inhale}, Exhales: {self.exhale}"