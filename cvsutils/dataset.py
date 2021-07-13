import io
import os
import random
import zipfile
import PIL.Image


class DatasetReader:
    @classmethod
    def open(cls, filename):
        dataset_type = cls.detect_type(filename)
        if dataset_type == 'object_detection':
            return cls.read_object_detection_dataset(filename)
        elif dataset_type == 'image_classification':
            return cls.read_image_classification_dataset(filename)
        else:
            raise RuntimeError

    @staticmethod
    def detect_type(filename):
        with open(filename) as f:
            for line in f:
                image, labels = line.strip().split()
                if '.' in labels:
                    return 'object_detection'
        return 'image_classification'

    @staticmethod
    def read_labels(filename, num_labels):
        labels_filename = os.path.join(os.path.dirname(filename), 'labels.txt')
        if os.path.exists(labels_filename):
            with open(labels_filename) as f:
                return [l.strip() for l in f]
        else:
            return [f'label_{i}' for i in range(num_labels)]

    @staticmethod
    def read_object_detection_dataset(filename):
        dataset = Dataset('object_detection', os.path.dirname(filename))
        reader = FileReader(os.path.dirname(filename))
        max_label = 0
        with open(filename) as f:
            for line in f:
                image, labels = line.strip().split()
                assert image and labels
                labels_file = reader.read(labels)
                # id, x, y, x2, y2
                labels = [[int(float(v)) for v in labels_line.strip().split()] for labels_line in labels_file]
                label_ids = [v[0] for v in labels]
                if label_ids:
                    max_label = max(max_label, *label_ids)
                dataset.add_data(image, labels)

        dataset.labels = DatasetReader.read_labels(filename, max_label + 1)
        dataset.validate()
        return dataset

    @staticmethod
    def read_image_classification_dataset(filename):
        dataset = Dataset('image_classification', os.path.dirname(filename))
        max_label = 0
        with open(filename) as f:
            for line in f:
                image, labels = line.strip().split()
                labels = [int(l) for l in labels.strip().split(',')]
                max_label = max(max_label, *labels)
                dataset.add_data(image, labels)

        dataset.labels = DatasetReader.read_labels(filename, max_label + 1)
        dataset.validate()
        return dataset


class FileReader:
    def __init__(self, base_dir):
        self.zip_objects = {}
        self.base_dir = base_dir

    def read(self, filepath, mode='r'):
        assert mode in ('r', 'rb')

        if '@' in filepath:
            zip_filepath, entrypath = filepath.split('@')
            if zip_filepath not in self.zip_objects:
                self.zip_objects[zip_filepath] = zipfile.ZipFile(os.path.join(self.base_dir, zip_filepath))
            with self.zip_objects[zip_filepath].open(entrypath) as f:
                return [line for line in f.read().decode('utf-8').split('\n') if line] if mode == 'r' else f.read()
        else:
            with open(os.path.join(self.base_dir, filepath), mode) as f:
                return f.read()


class Dataset:
    def __init__(self, dataset_type, base_dir='.'):
        assert dataset_type in ('image_classification', 'object_detection')

        self.base_dir = os.path.dirname(base_dir)
        self.dataset_type = dataset_type
        self.reader = FileReader(base_dir)
        self.images = []
        self.labels = []  # Optional label names.

    def validate(self):
        """Verify that the dataset is in valid state"""
        assert self.images
        if self.dataset_type == 'image_classification':
            pass
        elif self.dataset_type == 'object_detection':
            for i, (image, labels) in enumerate(self.images):
                if not image:
                    raise RuntimeError(f"{i}: missing an image.")
                for label, x, y, x2, y2 in labels:
                    if label < 0 or x <0 or y < 0 or x >= x2 or y >= y2:
                        raise RuntimeError(f"{i}: Invalid bounding box: {label} {x} {y} {x2} {y2}")

    def add_data(self, image, labels):
        assert image
        assert isinstance(labels, list) or (isinstance(labels, str) and labels)

        self.images.append((image, labels))

    def read_image(self, image_path):
        return self.reader.read(image_path, 'rb')

    def __len__(self):
        return len(self.images)

    def shuffle(self):
        random.shuffle(self.images)

    def get(self, index):
        image, labels = self.images[index]

        if isinstance(image, str):
            image = self.read_image(image)

        return (image, labels)


class DatasetWriter:
    @staticmethod
    def write(dataset, filename):
        dataset.validate()

        base_dir = os.path.dirname(filename)
        dataset.shuffle()

        images_zip_filename = 'images.zip'
        images_zip = zipfile.ZipFile(os.path.join(base_dir, images_zip_filename),
                                     mode='w', compression=zipfile.ZIP_STORED)
        if dataset.dataset_type == 'object_detection':
            labels_zip_filename = 'labels.zip'
            labels_zip = zipfile.ZipFile(os.path.join(base_dir, labels_zip_filename),
                                         mode='w', compression=zipfile.ZIP_STORED)

        with open(filename, 'w') as f:
            for i in range(len(dataset)):
                image, labels = dataset.get(i)
                ext = DatasetWriter.detect_imagetype(image)
                image_filepath = f'{i}.{ext}'
                images_zip.writestr(image_filepath, data=image)
                image_filepath = f'{images_zip_filename}@{image_filepath}'

                if dataset.dataset_type == 'object_detection':
                    labels_filepath = f'{i}.txt'
                    with labels_zip.open(labels_filepath, 'w') as lf:
                        for l in labels:
                            lf.write((' '.join([str(ls) for ls in l]) + '\n').encode('utf-8'))
                    labels = f'{labels_zip_filename}@{labels_filepath}'
                elif dataset.dataset_type == 'image_classification':
                    assert isinstance(labels, list)
                    assert len(labels) > 0 or isinstance(labels[0], int)
                    labels = ','.join([str(l) for l in labels])
                else:
                    raise NotImplementedError

                f.write(f'{image_filepath} {labels}\n')

        images_zip.close()
        if dataset.dataset_type == 'object_detection':
            labels_zip.close()

        if dataset.labels:
            with open(os.path.join(base_dir, 'labels.txt'), 'w') as f:
                for label_name in dataset.labels:
                    f.write(label_name + '\n')

    def detect_imagetype(image_binary):
        with PIL.Image.open(io.BytesIO(image_binary)) as img:
            image_format = img.format
        if image_format == 'JPEG':
            return 'jpg'
        elif image_format == 'BMP':
            return 'bmp'
        elif image_format == 'PNG':
            return 'png'
        else:
            raise NotImplementedError
