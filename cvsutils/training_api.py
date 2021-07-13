import urllib.parse
import uuid
import requests
import tenacity


class TrainingApi:
    CREATE_PROJECT_API = '/customvision/v3.2/training/projects'
    PROJECT_API = '/customvision/v3.2/training/projects/{project_id}'
    DOMAINS_API = '/customvision/v3.2/training/domains'

    CREATE_IMAGE_API = PROJECT_API + '/images'
    TAG_API = PROJECT_API + '/tags'
    TRAIN_PROJECT_API = PROJECT_API + '/train'
    TAGGED_IMAGES_API = PROJECT_API + '/images/tagged'
    TAGGED_IMAGES_COUNT_API = PROJECT_API + '/images/tagged/count'
    UNTAGGED_IMAGES_API = PROJECT_API + '/images/untagged'
    UNTAGGED_IMAGES_COUNT_API = PROJECT_API + '/images/untagged/count'
    SET_IMAGE_TAG_API = PROJECT_API + '/images/tags'
    SET_IMAGE_REGION_API = PROJECT_API + '/images/regions'

    ITERATIONS_API = PROJECT_API + '/iterations'
    ITERATION_API = PROJECT_API + '/iterations/{iteration_id}'
    ITERATION_EVAL_API = ITERATION_API + '/performance'
    ITERATION_PUBLISH_API = ITERATION_API + '/publish'
    EXPORT_API = ITERATION_API + '/export'
    DOMAIN_API = '/customvision/v3.2/training/domains/{domain_id}'

    def __init__(self, env):
        self.env = env
        self.api_url = env.training_endpoint
        self.training_key = env.training_key

    def train(self, project_id, force, domain_id=None, classification_type=None, export_capability=None):
        assert (not classification_type) or classification_type in ['multilabel', 'multiclass']
        export_capability = export_capability or []
        assert isinstance(export_capability, list)
        if domain_id or classification_type or export_capability:
            url = self.PROJECT_API.format(project_id=project_id)
            response = self._request('GET', url)
            current_domain_id = uuid.UUID(response['settings']['domainId'])
            updated = False
            if domain_id and current_domain_id != domain_id:
                response['settings']['domainId'] = str(domain_id)
                updated = True
            if classification_type and response['settings']['classificationType'] != classification_type:
                response['settings']['classificationType'] = classification_type
                updated = True
            if export_capability and set(response['settings']['targetExportPlatforms']) != set(export_capability):
                response['settings']['targetExportPlatforms'] = export_capability
                updated = True
            if updated:
                self._request('PATCH', url, json=response)

        url = self.TRAIN_PROJECT_API.format(project_id=project_id)
        params = {'forceTrain': force}
        response = self._request('POST', url, params)
        return uuid.UUID(response['id'])

    def create_project(self, project_name, domain_id=None):
        params = {'name': project_name}
        if domain_id:
            params['domainId'] = domain_id

        response = self._request('POST', self.CREATE_PROJECT_API, params)
        return uuid.UUID(response['id'])

    def create_image(self, project_id, image_binary):
        url = self.CREATE_IMAGE_API.format(project_id=project_id)
        response = self._request('POST', url, files={'files[0]': image_binary})
        return uuid.UUID(response['images'][0]['image']['id'])

    def create_images(self, project_id, image_binary_list):
        assert isinstance(project_id, uuid.UUID)
        assert isinstance(image_binary_list, list)

        url = self.CREATE_IMAGE_API.format(project_id=project_id)
        response = self._request('POST', url, files={str(i): binary for i, binary in enumerate(image_binary_list)})
        sorted_images = sorted(response['images'], key=lambda i: int(i['sourceUrl'].replace('"', '')))
        return [uuid.UUID(response_image['image']['id']) for response_image in sorted_images]

    def create_tag(self, project_id, tag_name):
        url = self.TAG_API.format(project_id=project_id)
        params = {'name': tag_name}
        response = self._request('POST', url, params)
        return uuid.UUID(response['id'])

    def export_iteration(self, project_id, iteration_id, platform, flavor):
        url = self.EXPORT_API.format(project_id=project_id, iteration_id=iteration_id)
        params = {'platform': platform}
        if flavor:
            params['flavor'] = flavor
        response = self._request('POST', url, params)
        return {'status': response['status']}  # TODO

    def get_exports(self, project_id, iteration_id, platform, flavor):
        url = self.EXPORT_API.format(project_id=project_id, iteration_id=iteration_id)
        response = self._request('GET', url)
        platform = platform.lower()
        flavor = flavor.lower() if flavor else None

        for entry in response:
            if entry['platform'].lower() == platform and ((entry['flavor'] is None and flavor is None) or (entry['flavor'] and entry['flavor'].lower() == flavor)):
                return {'status': entry['status'], 'url': entry['downloadUri']}
        return None

    def get_iteration(self, project_id, iteration_id):
        url = self.ITERATION_API.format(project_id=project_id, iteration_id=iteration_id)
        response = self._request('GET', url)
        if response['classificationType'] == 'Multiclass':
            task_type = 'multiclass_classification'
        elif response['classificationType'] == 'Multilabel':
            task_type = 'multilabel_classification'
        else:
            task_type = 'object_detection'

        return {'id': iteration_id,
                'project_id': project_id,
                'status': response['status'],
                'publish_name': response['publishName'],
                'domain_id': uuid.UUID(response['domainId']) if response['domainId'] else None,
                'task_type': task_type}

    def get_iterations(self, project_id):
        url = self.ITERATIONS_API.format(project_id=project_id)
        response = self._request('GET', url)
        iterations = []

        for r in response:
            domain_id = uuid.UUID(r['domainId']) if r['domainId'] else None
            iterations.append({'id': uuid.UUID(r['id']), 'name': r['name'], 'domain_id': domain_id, 'created_at': r['created'], 'publish_name': r['publishName']})
        return iterations

    def get_iteration_eval(self, project_id, iteration_id, threshold=0.5, iou_threshold=0.3):
        url = self.ITERATION_EVAL_API.format(project_id=project_id, iteration_id=iteration_id)
        params = {'threshold': threshold, 'overlapThreshold': iou_threshold}
        response = self._request('GET', url, params)
        return {'precision': response['precision'],
                'recall': response['recall'],
                'average_precision': response['averagePrecision']}

    def get_project(self, project_id):
        url = self.PROJECT_API.format(project_id=project_id)

        response = self._request('GET', url)
        return {
            'name': response['name'],
            'description': response['description'],
            'domain_id': uuid.UUID(response['settings']['domainId'])
        }

    def get_projects(self):
        response = self._request('GET', self.CREATE_PROJECT_API)
        projects = []
        for r in response:
            projects.append({'id': uuid.UUID(r['id']), 'name': r['name'], 'created_at': r['created'], 'modified_at': r['lastModified']})
        return projects

    def get_tags(self, project_id):
        """Get a list of pairs of (tag_name, tag_id). The returned list is sorted by tag_name."""
        url = self.TAG_API.format(project_id=project_id)
        response = self._request('GET', url)
        return [(t['name'], uuid.UUID(t['id'])) for t in response]

    def get_images(self, project_id):
        url = self.TAGGED_IMAGES_COUNT_API.format(project_id=project_id)
        num_tagged_images = self._request('GET', url)
        url = self.TAGGED_IMAGES_API.format(project_id=project_id)
        all_images = []

        def parse_labels(response):
            if 'regions' in response:
                return [[uuid.UUID(r['tagId']), r['left'], r['top'], r['left'] + r['width'], r['top'] + r['height']] for r in response['regions']]
            elif 'tags' in response:
                return [uuid.UUID(t['tagId']) for t in response['tags']]
            else:
                raise RuntimeError

        for i in range(((num_tagged_images-1)//256) + 1):
            params = {'take': 256, 'skip': 256 * i}
            response = self._request('GET', url, params)
            all_images.extend([{'url': r['originalImageUri'], 'labels': parse_labels(r)} for r in response])

        assert len(all_images) == num_tagged_images

        url = self.UNTAGGED_IMAGES_COUNT_API.format(project_id=project_id)
        num_untagged_images = self._request('GET', url)
        url = self.UNTAGGED_IMAGES_API.format(project_id=project_id)
        for i in range(((num_untagged_images-1)//256) + 1):
            params = {'take': 256, 'skip': 256 * i}
            response = self._request('GET', url, params)
            all_images.extend([{'url': r['originalImageUri'], 'labels': []} for r in response])
        assert len(all_images) == num_tagged_images + num_untagged_images
        return all_images

    def get_domain(self, domain_id):
        url = self.DOMAIN_API.format(domain_id=domain_id)

        response = self._request('GET', url)
        if response['type'] == 'Classification':
            domain_type = 'image_classification'
        elif response['type'] == 'ObjectDetection':
            domain_type = 'object_detection'
        else:
            raise RuntimeError(f"Unknown domain type: {response['type']}")

        return {
            'name': response['name'],
            'type': domain_type,
        }

    @staticmethod
    def _map_domain_type(domain_type):
        if domain_type == 'Classification':
            return 'image_classification'
        elif domain_type == 'ObjectDetection':
            return 'object_detection'
        else:
            raise RuntimeError(f"Unknown domain type: {domain_type}")

    def get_domains(self):
        response = self._request('GET', self.DOMAINS_API)
        return [{'id': r['id'], 'name': r['name'], 'type': self._map_domain_type(r['type'])} for r in response]

    def publish_iteration(self, project_id, iteration_id, publish_name):
        url = self.ITERATION_PUBLISH_API.format(project_id=project_id, iteration_id=iteration_id)
        params = {'publishName': publish_name,
                  'predictionId': self.env.prediction_resource_id}

        # This API doesn't have response body.
        self._request('POST', url, params=params, raw_response=True)

    def unpublish_iteration(self, project_id, iteration_id):
        url = self.ITERATION_PUBLISH_API.format(project_id=project_id, iteration_id=iteration_id)
        # This API doesn't have response body.
        self._request('DELETE', url, raw_response=True)

    def remove_iteration(self, project_id, iteration_id):
        url = self.ITERATION_API.format(project_id=project_id, iteration_id=iteration_id)
        self._request('DELETE', url)

    def set_image_classification_tags(self, project_id, image_tag_ids):
        assert isinstance(project_id, uuid.UUID)
        assert all(isinstance(t[0], uuid.UUID) for t in image_tag_ids)
        assert all(isinstance(t[1], uuid.UUID) for t in image_tag_ids)

        url = self.SET_IMAGE_TAG_API.format(project_id=project_id)
        tags = {'tags': [{'imageId': str(t[0]), 'tagId': str(t[1])} for t in image_tag_ids]}
        response = self._request('POST', url, json=tags)
        return len(response['created']) == len(image_tag_ids)

    def set_object_detection_tags(self, project_id, image_ids_labels):
        """
        image_ids_labels: [(image_id, labels), (image_id, labels), ...]
        """
        assert isinstance(project_id, uuid.UUID)
        assert all(isinstance(i[0], uuid.UUID) for i in image_ids_labels) and all(isinstance(i[1], list) for i in image_ids_labels)
        # assert max([i for label in image_ids_labels for i in label[1][1:]]) <= 1.0
        # assert min([i for label in image_ids_labels for i in label[1][1:]]) >= 0

        url = self.SET_IMAGE_REGION_API.format(project_id=project_id)
        regions = [{'imageId': str(i[0]), 'tagId': str(i[1][0]),
                    'left': max(0, i[1][1]), 'top': max(0, i[1][2]),
                    'width': min(1, i[1][3]) - max(0, i[1][1]), 'height': min(1, i[1][4]) - max(0, i[1][2])} for i in image_ids_labels]

        created = 0
        for i in range(int((len(regions)-0.5)//64)+1):
            batch = regions[i*64:(i+1)*64]
            response = self._request('POST', url, json={'regions': batch})
            created += len(response['created'])

        return created == len(image_ids_labels)

    def remove_project(self, project_id):
        raise NotImplementedError

    @tenacity.retry(retry=tenacity.retry_if_exception_type(IOError), stop=tenacity.stop_after_attempt(4))
    def _request(self, method, api_path, params=None, data=None, files=None, json=None, raw_response=False):
        assert method in ['GET', 'POST', 'PATCH', 'DELETE']

        url = urllib.parse.urljoin(self.api_url, api_path)
        headers = {'Training-Key': self.training_key}
        response = requests.request(method, url, params=params, data=data, json=json, files=files, headers=headers)
        if not response.ok:
            print(response.json())

        response.raise_for_status()
        if raw_response:
            return response

        return response.json() if method != 'DELETE' else None
