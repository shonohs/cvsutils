import argparse
import io
import uuid
import PIL
from tqdm import tqdm
from ..common import Environment, with_published
from ..dataset import DatasetReader
from ..evaluator import MulticlassClassificationEvaluator, MultilabelClassificationEvaluator, ObjectDetectionEvaluator
from ..training_api import TrainingApi
from ..prediction_api import PredictionApi

MAX_IMAGE_SIZE = 4194304


def evaluate_project(env, project_id, iteration_id, dataset_filename):
    training_api = TrainingApi(env)
    prediction_api = PredictionApi(env)
    dataset = DatasetReader.open(dataset_filename)

    iteration = training_api.get_iteration(project_id, iteration_id)
    cvs_labels = training_api.get_tags(project_id)

    label_names = sorted([l[0] for l in cvs_labels])
    if dataset.labels != label_names:
        print("WARNING: Label is different between dataset and cvs project.")
        print("dataset labels: " + str(dataset.labels))
        print("cvs project labels: " + str(label_names))

    with with_published(training_api, iteration) as publish_name:
        targets = []
        predictions = []
        for i in tqdm(range(len(dataset)), "Evaluating the project"):
            image, labels = dataset.get(i)
            if len(image) > MAX_IMAGE_SIZE:
                tqdm.write(f"Image {i}: image size is too big ({len(image)}). re-compressing...")
                image = _compress_image(image)
                if len(image) > MAX_IMAGE_SIZE:
                    tqdm.write(f"failed. skipping image {i}")

            pred = prediction_api.predict(project_id, dataset.dataset_type, publish_name, image)
            w, h = _get_image_size(image)
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


def _get_image_size(image):
    with PIL.Image.open(io.BytesIO(image)) as f:
        return f.size


def _compress_image(image):
    output = io.BytesIO()
    with PIL.Image.open(io.BytesIO(image)) as f:
        f.save(output, format='JPEG')
        return output.getvalue()


def main():
    parser = argparse.ArgumentParser("Evaluate a project with a validation dataset")
    parser.add_argument('--project_id', type=str, help="Project Id")
    parser.add_argument('--iteration_id', type=str, help="Iteration Id")
    parser.add_argument('dataset_filename', type=str, help="Dataset file path")

    args = parser.parse_args()
    evaluate_project(Environment(), uuid.UUID(args.project_id), uuid.UUID(args.iteration_id), args.dataset_filename)


if __name__ == '__main__':
    main()
