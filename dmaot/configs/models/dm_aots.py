import os
from .default import DefaultModelConfig

class ModelConfig(DefaultModelConfig):
    def __init__(self):
        super().__init__()
        self.MODEL_NAME = 'DM_AOTS'
        self.MODEL_ENGINE = 'dmaotengine'
        self.MODEL_LSTT_NUM = 2
