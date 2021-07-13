import contextlib
import os
import uuid


KNOWN_DOMAINS = {
    uuid.UUID('ee85a74c-405e-4adc-bb47-ffa8ca0c9f31'): 'image_classification',  # General
    uuid.UUID('c151d5b5-dd07-472a-acc8-15d29dea8518'): 'image_classification',  # Food
    uuid.UUID('ca455789-012d-4b50-9fec-5bb63841c793'): 'image_classification',  # Landmarks
    uuid.UUID('b30a91ae-e3c1-4f73-a81e-c270bff27c39'): 'image_classification',  # Retail
    uuid.UUID('45badf75-3591-4f26-a705-45678d3e9f5f'): 'image_classification',  # Adult
    uuid.UUID('0732100f-1a38-4e49-a514-c9b44c697ab5'): 'image_classification',  # General (compact)
    uuid.UUID('8882951b-82cd-4c32-970b-d5f8cb8bf6d7'): 'image_classification',  # Food (compact)
    uuid.UUID('b5cfd229-2ac7-4b2b-8d0a-2b0661344894'): 'image_classification',  # Landmarks (compact)
    uuid.UUID('6b4faeda-8396-481b-9f8b-177b9fa3097f'): 'image_classification',  # Retail (compact)
    uuid.UUID('a8e3c40f-fb4a-466f-832a-5e457ae4a344'): 'image_classification',  # General [A1]
    uuid.UUID('2e37d7fb-3a54-486a-b4d6-cfc369af0018'): 'image_classification',  # General [A2]
    uuid.UUID('da2e3a8a-40a5-4171-82f4-58522f70fbc1'): 'object_detection',  # General OD
    uuid.UUID('1d8ffafe-ec40-4fb2-8f90-72b3b6cecea4'): 'object_detection',  # Logo OD
    uuid.UUID('a27d5ca5-bb19-49d8-a70a-fec086c47f5b'): 'object_detection',  # General (compact) OD
    uuid.UUID('3780a898-81c3-4516-81ae-3a139614e1f3'): 'object_detection',
    uuid.UUID('7ec2ac80-887b-48a6-8df9-8b1357765430'): 'object_detection'
}


class Environment:
    DEFAULT_ENDPOINT = 'https://southcentralus.api.cognitive.microsoft.com/'

    def __init__(self):
        self._training_key = os.getenv('CVS_TRAINING_KEY', None)
        self._prediction_key = os.getenv('CVS_PREDICTION_KEY', None)

        endpoint = os.getenv('CVS_ENDPOINT', Environment.DEFAULT_ENDPOINT)
        self._training_endpoint = os.getenv('CVS_TRAINING_ENDPOINT', endpoint)
        self._prediction_endpoint = os.getenv('CVS_PREDICTION_ENDPOINT', endpoint)

        self._prediction_resource_id = os.getenv('CVS_PREDICTION_RESOURCE_ID', None)

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

    @property
    def prediction_endpoint(self):
        return self._prediction_endpoint

    @property
    def prediction_resource_id(self):
        if not self._prediction_resource_id:
            raise RuntimeError('Please set CVS_PREDICTION_RESOURCE_ID')
        return self._prediction_resource_id


def get_task_type_by_domain_id(domain_id):
    assert isinstance(domain_id, uuid.UUID)
    return KNOWN_DOMAINS.get(domain_id, None)


@contextlib.contextmanager
def with_published(training_api, iteration):
    publish_name = iteration['publish_name']
    published = False
    if not publish_name:
        publish_name = uuid.uuid4()
        training_api.publish_iteration(iteration['project_id'], iteration['id'], publish_name)
        published = True
        print(f"Published the iteration to {publish_name}")

    yield publish_name

    if published:
        training_api.unpublish_iteration(iteration['project_id'], iteration['id'])
        print("Unpublished the iteration")
