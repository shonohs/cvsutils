import argparse
import pathlib
import uuid
from ..common import Environment, get_task_type_by_domain_id, with_published
from ..prediction_api import PredictionApi
from ..training_api import TrainingApi


def predict_image(env, project_id, iteration_id, image_filepath, threshold):
    training_api = TrainingApi(env)
    prediction_api = PredictionApi(env)

    image = image_filepath.read_bytes()
    iteration = training_api.get_iteration(project_id, iteration_id)
    task_type = get_task_type_by_domain_id(iteration['domain_id'])

    with with_published(training_api, iteration) as publish_name:
        result = prediction_api.predict(project_id, task_type, publish_name, image)

    result = sorted(result, key=lambda r: r['probability'], reverse=True)

    for p in result:
        if p['probability'] > threshold:
            if task_type == 'image_classification':
                print(f"{p['label_name']:<16s}: {p['probability']:.3f}")
            else:
                print(f"{p['label_name']:<16s}: {p['probability']:.3f} box: ({p['left']:.2f}, {p['top']:.2f}, {p['right']:.2f}, {p['bottom']:.2f})")


def main():
    parser = argparse.ArgumentParser(description="Send a prediction request to Custom Vision Service")
    parser.add_argument('project_id', help="Project Id")
    parser.add_argument('iteration_id', help="Iteration Id")
    parser.add_argument('image_filepath', type=pathlib.Path, help="Image filename")
    parser.add_argument('--threshold', type=float, default=0, help="Probability threshold to show")

    args = parser.parse_args()

    if not args.image_filepath.exists():
        parser.error("f{args.image_filepath} is not found")

    predict_image(Environment(), uuid.UUID(args.project_id), uuid.UUID(args.iteration_id), args.image_filepath, args.threshold)


if __name__ == '__main__':
    main()
