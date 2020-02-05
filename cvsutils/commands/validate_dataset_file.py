import argparse
import io
import PIL
import tqdm
from ..dataset import DatasetReader


def validate_dataset_file(dataset_filename):
    dataset = DatasetReader.open(dataset_filename)
    dataset.validate()

    for i in tqdm.tqdm(range(len(dataset))):
        image, labels = dataset.get(i)
        size = PIL.Image.open(io.BytesIO(image)).size
        assert size[0] > 0 and size[1] > 0


def main():
    parser = argparse.ArgumentParser("Validate a dataset file")
    parser.add_argument('dataset_filename', type=str, help="Dataset file path")

    args = parser.parse_args()
    validate_dataset_file(args.dataset_filename)


if __name__ == '__main__':
    main()
