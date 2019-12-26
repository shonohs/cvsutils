import os


class Environment:
    DEFAULT_TRAINING_ENDPOINT = 'https://southcentralus.api.cognitive.microsoft.com/'

    def __init__(self):
        self._training_key = os.getenv('CVS_TRAINING_KEY', None)
        self._prediction_key = os.getenv('CVS_PREDICTION_KEY', None)
        self._training_endpoint = os.getenv('CVS_TRAINING_ENDPOINT', Environment.DEFAULT_TRAINING_ENDPOINT)

    @property
    def training_key(self):
        if not self._training_key:
            raise RuntimeError('Please set CVS_TRAINING_KEY')
        return self._training_key

    @property
    def prediction_key(self):
        if not self._prediction_key:
            raise RuntimeError('Please set CVS_PREDICTION_KEY')
        return self._prediction_key

    @property
    def training_endpoint(self):
        return self._training_endpoint
