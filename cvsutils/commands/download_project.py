import argparse
import os
import requests
import uuid
from tqdm import tqdm
from ..common import Environment
from ..dataset import Dataset, DatasetWriter
from ..training_api import TrainingApi


def download_project(env, project_id, dataset_filename):
    if os.path.exists(dataset_filename):
        raise RuntimeError(f"{dataset_filename} already exists")

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
            labels = [[tag_ids.index(t[0]), *t[1:]] for t in entry['labels']]
        else:
            raise RuntimeError

        dataset.add_data(image, labels)

    print(f"Downloaded {len(dataset)} images")

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
