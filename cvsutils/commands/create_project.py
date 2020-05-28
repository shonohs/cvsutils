import argparse
import io
import os
import uuid
import PIL
from tqdm import tqdm
from ..common import Environment
from ..dataset import DatasetReader
from ..training_api import TrainingApi

DEFAULT_IC_DOMAIN_ID = 'ee85a74c-405e-4adc-bb47-ffa8ca0c9f31'
DEFAULT_OD_DOMAIN_ID = 'da2e3a8a-40a5-4171-82f4-58522f70fbc1'
BATCH_SIZE = 32


def create_project(env, dataset_filename, project_name, domain_id):
    training_api = TrainingApi(env)
    dataset = DatasetReader.open(dataset_filename)

    # Set default project name. {dir_name}/{file_name}
    if not project_name:
        dir_name = os.path.basename(os.path.dirname(dataset_filename))
        file_name = os.path.basename(dataset_filename)
        project_name = f'{dir_name}/{file_name}'

    # Set default domain id.
    if domain_id:
        domain_id = uuid.UUID(domain_id)
    else:
        if dataset.dataset_type == 'image_classification':
            domain_id = uuid.UUID(DEFAULT_IC_DOMAIN_ID)
        elif dataset.dataset_type == 'object_detection':
            domain_id = uuid.UUID(DEFAULT_OD_DOMAIN_ID)

    # Create a project
    project_id = training_api.create_project(project_name, domain_id)
    print(f"Created project: {project_id}")

    # Create tags
    tag_ids = []
    for tag_name in dataset.labels:
        tag_ids.append(training_api.create_tag(project_id, tag_name))
    print(f"Created {len(tag_ids)} tags.")

    # Upload images
    batch_images = []
    batch_labels = []
    for i in tqdm(range(len(dataset)), "Uploading images"):
        image, labels = dataset.get(i)
        batch_images.append(image)
        batch_labels.append(labels)

        if len(batch_images) >= BATCH_SIZE:
            _upload_batch(training_api, dataset.dataset_type, project_id, tag_ids, batch_images, batch_labels)
            batch_images = []
            batch_labels = []

    if batch_images:
        _upload_batch(training_api, dataset.dataset_type, project_id, tag_ids, batch_images, batch_labels)

    print(f"Uploaded {len(dataset)} images")


def _get_image_size(image):
    with PIL.Image.open(io.BytesIO(image)) as f:
        return f.size


def _upload_batch(training_api, dataset_type, project_id, tag_ids, batch_images, batch_labels):
    image_ids = training_api.create_images(project_id, batch_images)
    if dataset_type == 'image_classification':
        labels = [(image_ids[image_index], tag_ids[l]) for image_index, labels in enumerate(batch_labels) for l in labels]
        if labels:
            training_api.set_image_classification_tags(project_id, labels)
    elif dataset_type == 'object_detection':
        image_sizes = [_get_image_size(i) for i in batch_images]
        labels = [(image_ids[image_index], [tag_ids[l[0]],
                                            l[1] / image_sizes[image_index][0],
                                            l[2] / image_sizes[image_index][1],
                                            l[3] / image_sizes[image_index][0],
                                            l[4] / image_sizes[image_index][1]]) for image_index, labels in enumerate(batch_labels) for l in labels]

        if labels:
            training_api.set_object_detection_tags(project_id, labels)
    else:
        raise RuntimeError(f"Unknown dataset type: {dataset_type}")


def main():
    parser = argparse.ArgumentParser("Upload a project to Custom Vision Service")
    parser.add_argument('dataset_filename', type=str, help="Dataset file path")
    parser.add_argument('--project_name', type=str, default=None, help="Project name")
    parser.add_argument('--domain_id', type=str, default=None, help="Domain id")

    args = parser.parse_args()
    create_project(Environment(), args.dataset_filename, args.project_name, args.domain_id)


if __name__ == '__main__':
    main()
