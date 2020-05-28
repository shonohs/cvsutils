import argparse
import uuid
from ..common import Environment
from ..training_api import TrainingApi


def remove_iteration(env, project_id, iteration_id):
    training_api = TrainingApi(env)

    training_api.remove_iteration(project_id, iteration_id)

    print(f"Removed iteration {iteration_id}")


def main():
    parser = argparse.ArgumentParser("Remove an iteration")
    parser.add_argument('project_id', type=str)
    parser.add_argument('iteration_id', type=str)

    args = parser.parse_args()

    remove_iteration(Environment(), uuid.UUID(args.project_id), uuid.UUID(args.iteration_id))


if __name__ == '__main__':
    main()
