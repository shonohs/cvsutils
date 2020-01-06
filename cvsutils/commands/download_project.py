import argparse
import io
import os
import requests
import uuid
import PIL
from tqdm import tqdm
from ..common import Environment
from ..dataset import Dataset, DatasetWriter
from ..training_api import TrainingApi


def download_project(env, project_id, dataset_filename):
    if os.path.exists(dataset_filename):
        raise RuntimeError(f"{dataset_filename} already exists")
    if os.path.exists(os.path.join(os.path.dirname(dataset_filename), 'images.zip')):
        raise RuntimeError("images.zip already exists")
    if os.path.exists(os.path.join(os.path.dirname(dataset_filename), 'labels.zip')):
        raise RuntimeError("labels.zip already exists")
    if os.path.exists(os.path.join(os.path.dirname(dataset_filename), 'labels.txt')):
        raise RuntimeError("labels.txt already exists")

    training_api = TrainingApi(env.training_endpoint, env.training_key)
    domain_id = training_api.get_project(project_id)['domain_id']
    domain_type = training_api.get_domain(domain_id)['type']
    dataset = Dataset(domain_type, os.path.dirname(dataset_filename))

    tags = training_api.get_tags(project_id)
    tag_names, tag_ids = zip(*tags)
    dataset.labels = tag_names

    images = training_api.get_images(project_id)
    print(f"Found {len(images)} images")

    for entry in tqdm(images, "Downloading images"):
        # Download image
        response = requests.get(entry['url'])
        response.raise_for_status()
        image = response.content

        if domain_type == 'image_classification':
            labels = [tag_ids.index(t) for t in entry['labels']]
        elif domain_type == 'object_detection':
            image_size = PIL.Image.open(io.BytesIO(image)).size
            labels = [[tag_ids.index(t[0]),
                       int(t[1] * image_size[0]),
                       int(t[2] * image_size[1]),
                       int(t[3] * image_size[0]),
                       int(t[4] * image_size[1])] for t in entry['labels']]
        else:
            raise RuntimeError

        dataset.add_data(image, labels)

    print(f"Downloaded {len(dataset)} images")

    os.makedirs(os.path.dirname(dataset_filename), exist_ok=True)
    DatasetWriter.write(dataset, dataset_filename)
    print(f"Saved the dataset to {dataset_filename}")


def main():
    parser = argparse.ArgumentParser("Download a project from Custom Vision Service")
    parser.add_argument('project_id', type=str, help="Project id")
    parser.add_argument('dataset_filename', type=str, help="Dataset file path")

    args = parser.parse_args()
    download_project(Environment(), uuid.UUID(args.project_id), args.dataset_filename)


if __name__ == '__main__':
    main()
