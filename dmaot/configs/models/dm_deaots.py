from .default_deaot import DefaultModelConfig


class ModelConfig(DefaultModelConfig):
    def __init__(self):
        super().__init__()
        self.MODEL_NAME = 'DM_DeAOTS'
        self.MODEL_ENGINE = 'dmdeaotengine'

        self.MODEL_LSTT_NUM = 2
