import argparse
import uuid
from ..common import Environment
from ..training_api import TrainingApi


def show_project(env, project_id):
    training_api = TrainingApi(env)

    project = training_api.get_project(project_id)
    num_images = training_api.get_num_images(project_id)

    print(f"Name: {project['name']}")
    print(f"Description: {project['description']}")
    print(f"Domain: {project['domain_id']}")
    print(f"Created at: {project['created_at']}")
    print(f"The number of images: {num_images}")
    print("")

    tags = training_api.get_tags(project_id)
    print("Tags:")
    for tag_name, tag_id in tags:
        print(f"    {tag_name} ({tag_id})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('project_id', type=uuid.UUID)

    args = parser.parse_args()
    show_project(Environment(), args.project_id)


if __name__ == '__main__':
    main()
