
class ObjectDetectionDataset:
    def  __init__(self, filename):
        self.images = []
        with open(filename) as f:
            for line in f:
                image_filepath, label_filepath = line.strip().split()
                image_filepath = image_filepath.strip()
                label_filepath = label_filepath.strip()
                self.images.append((image_filepath, label_filepath))

    def add_data(self, image, labels):
        pass

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        ret = self.images[self.index]
        self.index += 1
        return ret
