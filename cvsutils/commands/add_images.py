import argparse
import io
import pathlib
import uuid
import PIL.Image
import tqdm
from ..common import Environment
from ..dataset import DatasetReader
from ..training_api import TrainingApi


def _get_image_size(image):
    with PIL.Image.open(io.BytesIO(image)) as f:
        return f.size


def add_images(env, project_id, dataset_filepath):
    training_api = TrainingApi(env)
    dataset = DatasetReader.open(dataset_filepath)

    existing_tags = training_api.get_tags(project_id)
    existing_tag_ids = {x[0]: x[1] for x in existing_tags}  # Name => ID
    tags_to_be_added = [x for x in dataset.labels if x not in existing_tag_ids]
    print(f"New tags will be added: {tags_to_be_added}")
    response = input("Continue? [y/N]")
    if response.lower() != 'y':
        return

    tag_ids = []
    for new_tag_name in dataset.labels:
        if new_tag_name in existing_tag_ids:
            tag_id = existing_tag_ids[new_tag_name]
        else:
            tag_id = training_api.create_tag(project_id, new_tag_name)
        tag_ids.append(tag_id)

    for i in tqdm.tqdm(range(len(dataset)), "Uploading images"):
        image, labels = dataset.get(i)
        image_id = training_api.create_image(project_id, image)

        if dataset.dataset_type == 'image_classification':
            labels = [(image_id, tag_ids[label]) for label in labels]
            training_api.set_image_classification_tags(labels)
        elif dataset.dataset_type == 'object_detection':
            image_size = _get_image_size(image)
            labels = [(image_id, [tag_ids[label[0]], label[1] / image_size[0], label[2] / image_size[1], label[3] / image_size[0], label[4] / image_size[1]]) for label in labels]
            training_api.set_object_detection_tags(project_id, labels)

    print(f"Uploaded {len(dataset)} images.")


def main():
    parser = argparse.ArgumentParser(description="Add images to an existing project.")
    parser.add_argument('project_id', type=uuid.UUID)
    parser.add_argument('dataset_filepath', type=pathlib.Path)

    args = parser.parse_args()

    add_images(Environment(), args.project_id, args.dataset_filepath)


if __name__ == '__main__':
    main()
