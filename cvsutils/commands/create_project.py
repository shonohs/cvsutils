import argparse
import io
import pathlib
import uuid
import PIL
from tqdm import tqdm
from ..common import Environment
from ..dataset import DatasetReader
from ..training_api import TrainingApi

DEFAULT_IC_DOMAIN_ID = 'ee85a74c-405e-4adc-bb47-ffa8ca0c9f31'
DEFAULT_OD_DOMAIN_ID = 'da2e3a8a-40a5-4171-82f4-58522f70fbc1'


def create_project(env, dataset_filepath, project_name, domain_id, batch_size, ignore_error):
    training_api = TrainingApi(env)
    dataset = DatasetReader.open(dataset_filepath)

    # Set default project name. {dir_name}/{file_name}
    if not project_name:
        dir_name = dataset_filepath.resolve().parent.name
        file_name = dataset_filepath.name
        project_name = f'{dir_name}/{file_name}'

    # Set default domain id.
    if not domain_id:
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
    batch_indices = []
    for i in tqdm(range(len(dataset)), "Uploading images"):
        image, labels = dataset.get(i)
        batch_images.append(image)
        batch_labels.append(labels)
        batch_indices.append(i)

        if len(batch_images) >= batch_size:
            _upload_batch(training_api, dataset.dataset_type, project_id, tag_ids, batch_images, batch_labels, batch_indices, ignore_error)
            batch_images = []
            batch_labels = []
            batch_indices = []

    if batch_images:
        _upload_batch(training_api, dataset.dataset_type, project_id, tag_ids, batch_images, batch_labels, batch_indices, ignore_error)

    print(f"Uploaded {len(dataset)} images")


def _get_image_size(image):
    with PIL.Image.open(io.BytesIO(image)) as f:
        return f.size


def _upload_batch(training_api, dataset_type, project_id, tag_ids, batch_images, batch_labels, batch_indices, ignore_error):
    assert dataset_type in ['image_classification', 'object_detection']
    try:
        image_ids = training_api.create_images(project_id, batch_images)
    except Exception as e:
        print(f"Failed to upload images: {batch_indices}")
        print(e)
        if not ignore_error:
            raise

    try:
        if dataset_type == 'image_classification':
            labels = [(image_ids[image_index], tag_ids[label]) for image_index, labels in enumerate(batch_labels) for label in labels]
            if labels:
                training_api.set_image_classification_tags(project_id, labels)
        elif dataset_type == 'object_detection':
            image_sizes = [_get_image_size(i) for i in batch_images]
            labels = [(image_ids[image_index], [tag_ids[label[0]],
                                                label[1] / image_sizes[image_index][0],
                                                label[2] / image_sizes[image_index][1],
                                                label[3] / image_sizes[image_index][0],
                                                label[4] / image_sizes[image_index][1]]) for image_index, labels in enumerate(batch_labels) for label in labels]

            if labels:
                training_api.set_object_detection_tags(project_id, labels)
    except Exception as e:
        print(f"Failed to upload tags for {batch_indices}.")
        print(e)
        if not ignore_error:
            raise


def main():
    parser = argparse.ArgumentParser(description="Upload a project to Custom Vision Service")
    parser.add_argument('dataset_filename', type=pathlib.Path, help="Dataset file path")
    parser.add_argument('--project_name', help="Project name")
    parser.add_argument('--domain_id', type=uuid.UUID, help="Domain id")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--ignore_error', action='store_true')

    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("Batch size must be a positive number.")

    create_project(Environment(), args.dataset_filename, args.project_name, args.domain_id, args.batch_size, args.ignore_error)


if __name__ == '__main__':
    main()
