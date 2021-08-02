import argparse
import pathlib
import uuid
from tqdm import tqdm
from ..common import Environment, with_published, compress_image_if_needed_for_prediction, get_image_size
from ..dataset import DatasetReader
from ..evaluator import MulticlassClassificationEvaluator, MultilabelClassificationEvaluator, ObjectDetectionEvaluator
from ..training_api import TrainingApi
from ..prediction_api import PredictionApi


def evaluate_project(env, project_id, iteration_id, dataset_filename):
    training_api = TrainingApi(env)
    prediction_api = PredictionApi(env)
    dataset = DatasetReader.open(dataset_filename)

    iteration = training_api.get_iteration(project_id, iteration_id)
    cvs_labels = training_api.get_tags(project_id)

    label_names = sorted([label[0] for label in cvs_labels])
    if dataset.labels != label_names:
        print("WARNING: Label is different between dataset and cvs project.")
        print("dataset labels: " + str(dataset.labels))
        print("cvs project labels: " + str(label_names))

    with with_published(training_api, iteration) as publish_name:
        targets = []
        predictions = []
        for i in tqdm(range(len(dataset)), "Evaluating the project"):
            image, labels = dataset.get(i)
            image = compress_image_if_needed_for_prediction(image)
            pred = prediction_api.predict(project_id, dataset.dataset_type, publish_name, image)
            w, h = get_image_size(image)
            predictions.append([[label_names.index(p['label_name']), p['probability'], p['left'] * w, p['top'] * h, p['right'] * w, p['bottom'] * h] for p in pred])
            targets.append(labels)

    evaluator = _get_evaluator(iteration)
    evaluator.add_predictions(predictions, targets)
    report = evaluator.get_report()
    print(report)


def _get_evaluator(iteration):
    if iteration['task_type'] == 'multiclass_classification':
        return MulticlassClassificationEvaluator()
    elif iteration['task_type'] == 'multilabel_classification':
        return MultilabelClassificationEvaluator()
    elif iteration['task_type'] == 'object_detection':
        return ObjectDetectionEvaluator()


def main():
    parser = argparse.ArgumentParser("Evaluate a project with a validation dataset")
    parser.add_argument('--project_id', type=uuid.UUID, help="Project Id")
    parser.add_argument('--iteration_id', type=uuid.UUID, help="Iteration Id")
    parser.add_argument('dataset_filename', type=pathlib.Path, help="Dataset file path")

    args = parser.parse_args()
    evaluate_project(Environment(), args.project_id, args.iteration_id, args.dataset_filename)


if __name__ == '__main__':
    main()
