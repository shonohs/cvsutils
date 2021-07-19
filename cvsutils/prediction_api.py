import urllib
import uuid

import requests
import tenacity


class PredictionApi:
    CLASSIFY_IMAGE = '/customvision/v3.0/prediction/{project_id}/classify/iterations/{name}/image/nostore'
    DETECT_IMAGE = '/customvision/v3.0/prediction/{project_id}/detect/iterations/{name}/image/nostore'

    def __init__(self, env):
        self.api_url = env.prediction_endpoint
        self._session = requests.Session()
        self._session.headers.update({'Prediction-Key': env.prediction_key, 'Content-Type': 'application/octet-stream'})

    def predict(self, project_id, task_type, name, image_binary):
        assert task_type in ['image_classification', 'object_detection']

        url = self.CLASSIFY_IMAGE if task_type == 'image_classification' else self.DETECT_IMAGE
        url = url.format(project_id=project_id, name=name)

        response = self._request(url, data=image_binary)
        if task_type == 'object_detection':
            return [
                {'label_id': uuid.UUID(r['tagId']),
                 'label_name': r['tagName'],
                 'probability': r['probability'],
                 'left': r['boundingBox']['left'],
                 'top': r['boundingBox']['top'],
                 'right': r['boundingBox']['left'] + r['boundingBox']['width'],
                 'bottom': r['boundingBox']['top'] + r['boundingBox']['height']} for r in response['predictions']]
        else:
            return [
                {'label_id': uuid.UUID(r['tagId']),
                 'label_name': r['tagName'],
                 'probability': r['probability']} for r in response['predictions']]

    @tenacity.retry(retry=tenacity.retry_if_exception_type(IOError), stop=tenacity.stop_after_attempt(4), wait=tenacity.wait_exponential())
    def _request(self, api_path, data):
        url = urllib.parse.urljoin(self.api_url, api_path)
        response = self._session.request('POST', url, data=data, timeout=60)
        if not response.ok:
            print(response)

        response.raise_for_status()
        return response.json()
