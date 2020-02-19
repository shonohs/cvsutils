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


def download_project(env, project_id, output_directory):
    if os.path.exists(output_directory):
        raise RuntimeError(f"{output_directory} already exists")

    training_api = TrainingApi(env)
    domain_id = training_api.get_project(project_id)['domain_id']
    domain_type = training_api.get_domain(domain_id)['type']
    dataset = Dataset(domain_type, output_directory)

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

    os.makedirs(output_directory, exist_ok=True)
    DatasetWriter.write(dataset, os.path.join(output_directory, 'images.txt'))
    print(f"Saved the dataset to {output_directory}")


def main():
    parser = argparse.ArgumentParser("Download a project from Custom Vision Service")
    parser.add_argument('project_id', type=str, help="Project id")
    parser.add_argument('output_directory', type=str, help="Directory name for the downloaded files")

    args = parser.parse_args()
    download_project(Environment(), uuid.UUID(args.project_id), args.output_directory)


if __name__ == '__main__':
    main()
