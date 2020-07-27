import argparse
import time
import uuid
from ..common import Environment
from ..training_api import TrainingApi


def train_project(env, project_id, force, domain_id, classification_type, export_capability):
    training_api = TrainingApi(env)

    iteration_id = training_api.train(project_id, force, domain_id, classification_type, export_capability)
    print(f"Training started: iteration_id={iteration_id}")

    status = 'Training'
    start_time = time.time()
    while status == 'Training':
        status = training_api.get_iteration(project_id, iteration_id)['status']
        time.sleep(5)

    print(f"Training completed: {time.time()-start_time}s")

    results = training_api.get_iteration_eval(project_id, iteration_id)
    print(f"Average Precision={results['average_precision']}, Precision={results['precision']}, Recall={results['recall']}")


def main():
    parser = argparse.ArgumentParser("Train a project")
    parser.add_argument('project_id', type=str, help="Project id")
    parser.add_argument('--domain_id', type=str, default=None, help="Domain id")
    parser.add_argument('--force', action='store_true', help="Trigger training even if the dataset is not changed")
    parser.add_argument('--type', choices=['multiclass', 'multilabel'], default=None, help="Classification type")
    parser.add_argument('--capability', nargs='+', help="Export capability")

    args = parser.parse_args()
    train_project(Environment(), uuid.UUID(args.project_id), args.force, args.domain_id, args.type, args.capability)


if __name__ == '__main__':
    main()
