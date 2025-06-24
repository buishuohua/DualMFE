class NoExperimentFound(Exception):
    def __init__(self, message="No experiment found"):
        self.message = message
        super().__init__(self.message)

class NoCheckpointFound(Exception):
    def __init__(self, message="No checkpoint found"):
        self.message = message
        super().__init__(self.message)

class ModelNotBuilt(Exception):
    def __init__(self, message="Model not built"):
        self.message = message
        super().__init__(self.message)


class ModeNotSupported(Exception):
    def __init__(self, message="Mode not supported"):
        self.message = message
        super().__init__(self.message)


class ModelNotSupported(Exception):
    def __init__(self, message="Model not supported"):
        self.message = message
        super().__init__(self.message)
