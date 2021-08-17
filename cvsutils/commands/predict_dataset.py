import argparse
from collections import defaultdict
import pathlib
import uuid
import tqdm
from ..common import Environment, with_published, compress_image_if_needed_for_prediction, get_image_size
from ..dataset import DatasetReader, DatasetWriter, Dataset
from ..prediction_api import PredictionApi
from ..training_api import TrainingApi


def predict_dataset(env, project_id, iteration_id, input_dataset_filepath, output_dataset_filepath, prob_thresholds_per_label):
    training_api = TrainingApi(env)
    prediction_api = PredictionApi(env)

    iteration = training_api.get_iteration(project_id, iteration_id)
    domain_type = 'object_detection' if iteration['task_type'] == 'object_detection' else 'image_classification'
    cvs_labels = training_api.get_tags(project_id, iteration_id)

    dataset = DatasetReader.open(input_dataset_filepath)
    new_dataset = Dataset(domain_type, output_dataset_filepath.parent)
    tag_names, tag_ids = zip(*cvs_labels)
    new_dataset.labels = tag_names

    with with_published(training_api, iteration) as publish_name:
        for i in tqdm.tqdm(range(len(dataset)), "Predicting"):
            original_image_binary, _ = dataset.get(i)
            image_binary = compress_image_if_needed_for_prediction(original_image_binary)
            width, height = get_image_size(image_binary)
            pred = prediction_api.predict(project_id, dataset.dataset_type, publish_name, image_binary)
            pred = [p for p in pred if p['probability'] > prob_thresholds_per_label[p['label_name']]]
            if domain_type == 'image_classification':
                labels = [tag_ids.index(p['label_id']) for p in pred]
            elif domain_type == 'object_detection':
                labels = [[tag_ids.index(p['label_id']), int(p['left'] * width), int(p['top'] * height), int(p['right'] * width), int(p['bottom'] * height)] for p in pred]
            else:
                raise RuntimeError

            new_dataset.add_data(original_image_binary, labels)

    output_dataset_filepath.parent.mkdir(parents=True)
    DatasetWriter.write(new_dataset, output_dataset_filepath)
    print(f"Successfully saved the prediction results to {output_dataset_filepath}")


def main():
    parser = argparse.ArgumentParser(description="Perform predictions on the given dataset.")
    parser.add_argument('project_id', type=uuid.UUID)
    parser.add_argument('iteration_id', type=uuid.UUID)
    parser.add_argument('input_dataset_filepath', type=pathlib.Path, help="A dataset that contains input images. The labels will be ignored.")
    parser.add_argument('output_directory', type=pathlib.Path)
    parser.add_argument('--threshold', type=float, default=0.1, help="Probability threshold (default=0.1)")
    parser.add_argument('--threshold_per_label', default=[], nargs=2, metavar=('LABEL_NAME', 'THRESHOLD'), action='append', help="Probability threshold per label")

    args = parser.parse_args()

    if not args.input_dataset_filepath.exists():
        parser.error(f"{args.input_dataset_filepath} is not found.")

    if args.input_dataset_filepath.is_dir():
        parser.error(f"{args.input_dataset_filepath} is a directory path. Please specify a path for a txt file.")

    if args.output_directory.exists():
        parser.error(f"{args.output_directory} already exists.")

    if not (0 <= args.threshold <= 1):
        parser.error(f"Threshold must be in range [0, 1]. threshold={args.threshold}")

    prob_thresholds_per_label = defaultdict(lambda: args.threshold)
    for label_name, threshold in args.threshold_per_label:
        prob_thresholds_per_label[label_name] = float(threshold)

    output_dataset_filepath = args.output_directory / 'images.txt'
    predict_dataset(Environment(), args.project_id, args.iteration_id, args.input_dataset_filepath, output_dataset_filepath, prob_thresholds_per_label)


if __name__ == '__main__':
    main()
