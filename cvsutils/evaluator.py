from abc import ABC, abstractmethod
import collections
import statistics

import numpy as np
import sklearn.metrics
import torch


class Evaluator(ABC):
    """Class to evaluate model outputs and report the result.
    """
    def __init__(self):
        self.reset()

    @abstractmethod
    def add_predictions(self, predictions, targets):
        pass

    @abstractmethod
    def get_report(self):
        pass

    @abstractmethod
    def reset(self):
        pass


class MulticlassClassificationEvaluator(Evaluator):
    def add_predictions(self, predictions, targets):
        """ Evaluate a batch of predictions.
        Args:
            predictions: the model output tensor. Shape (N, num_class)
            targets: the golden truths. Shape (N,)
        """
        assert len(predictions) == len(targets)
        assert len(targets.shape) == 1

        # top-1 accuracy
        _, indices = torch.topk(predictions, 1)
        correct = indices.view(-1).eq(targets)
        correct_num = int(correct.long().sum(0))
        self.top1_correct_num += correct_num

        # top-5 accuracy
        k = min(5, predictions.shape[1])
        _, indices = torch.topk(predictions, k)
        correct = indices == targets.view(-1, 1).long().expand(-1, k)
        self.top5_correct_num += int(correct.long().sum())

        # Average precision
        target_vec = torch.zeros_like(predictions, dtype=torch.uint8)
        for i, t in enumerate(targets):
            target_vec[i, t] = 1
        ap = sklearn.metrics.average_precision_score(target_vec.view(-1).cpu().numpy(), predictions.view(-1).cpu().numpy(), average='macro')
        self.ap += ap * len(predictions)
        self.total_num += len(predictions)

    def get_report(self):
        return {'top1_accuracy': float(self.top1_correct_num) / self.total_num if self.total_num else 0.0,
                'top5_accuracy': float(self.top5_correct_num) / self.total_num if self.total_num else 0.0,
                'average_precision': self.ap / self.total_num if self.total_num else 0.0}

    def reset(self):
        self.top1_correct_num = 0
        self.top5_correct_num = 0
        self.ap = 0
        self.total_num = 0


class MultilabelClassificationEvaluator(Evaluator):
    def add_predictions(self, predictions, targets):
        """ Evaluate a batch of predictions.
        Args:
            predictions: the model output tensor. Shape (N, num_class)
            targets: the golden truths. Shape (N, num_class)
        """
        assert len(predictions) == len(targets)
        targets = targets.to(torch.uint8)
        num = torch.mul(predictions > 0.5, targets).long().sum(1)  # shape (N,)
        den = torch.add(predictions > 0.5, targets).ge(1).long().sum(1)  # shape (N,)
        den[den == 0] = 1  # To avoid zero-division. If den==0, num should be zero as well.
        self.correct_num += torch.sum(num.to(torch.float32) / den.to(torch.float32))

        ap = sklearn.metrics.average_precision_score(targets.view(-1).cpu().numpy(), predictions.view(-1).cpu().numpy(), average='macro')
        self.ap += ap * len(predictions)
        self.total_num += len(predictions)

    def get_report(self):
        return {'accuracy_50': float(self.correct_num) / self.total_num if self.total_num else 0.0,
                'average_precision': self.ap / self.total_num if self.total_num else 0.0}

    def reset(self):
        self.correct_num = 0
        self.ap = 0
        self.total_num = 0


class ObjectDetectionSingleIOUEvaluator(Evaluator):
    def __init__(self, iou):
        super(ObjectDetectionSingleIOUEvaluator, self).__init__()
        self.iou = iou

    def add_predictions(self, predictions, targets):
        """ Evaluate list of image with object detection results using single IOU evaluation.
        Args:
            predictions: list of predictions [[[label_idx, probability, L, T, R, B], ...], [...], ...]
            targets: list of image targets [[[label_idx, L, T, R, B], ...], ...]
        """

        assert len(predictions) == len(targets)

        eval_predictions = collections.defaultdict(list)
        eval_ground_truths = collections.defaultdict(dict)
        for img_idx, prediction in enumerate(predictions):
            for bbox in prediction:
                label = int(bbox[0])
                eval_predictions[label].append([img_idx, float(bbox[1]), float(bbox[2]), float(bbox[3]), float(bbox[4]), float(bbox[5])])

        for img_idx, target in enumerate(targets):
            for bbox in target:
                label = int(bbox[0])
                if img_idx not in eval_ground_truths[label]:
                    eval_ground_truths[label][img_idx] = []
                eval_ground_truths[label][img_idx].append([float(bbox[1]), float(bbox[2]), float(bbox[3]), float(bbox[4])])

        class_indices = set(list(eval_predictions.keys()) + list(eval_ground_truths.keys()))
        for class_index in class_indices:
            is_correct, probabilities = self._evaluate_predictions(eval_ground_truths[class_index], eval_predictions[class_index], self.iou)
            true_num = sum([len(l) for l in eval_ground_truths[class_index].values()])

            self.is_correct[class_index].extend(is_correct)
            self.probabilities[class_index].extend(probabilities)
            self.true_num[class_index] += true_num

    def _calculate_area(self, rect):
        w = rect[2] - rect[0]+1e-5
        h = rect[3] - rect[1]+1e-5
        return float(w * h) if w > 0 and h > 0 else 0.0

    def _calculate_iou(self, rect0, rect1):
        rect_intersect = [max(rect0[0], rect1[0]),
                          max(rect0[1], rect1[1]),
                          min(rect0[2], rect1[2]),
                          min(rect0[3], rect1[3])]
        area_intersect = self._calculate_area(rect_intersect)
        return area_intersect / (self._calculate_area(rect0) + self._calculate_area(rect1) - area_intersect)

    def _is_true_positive(self, prediction, ground_truth, already_detected, iou_threshold):
        image_id = prediction[0]
        prediction_rect = prediction[2:6]
        if image_id not in ground_truth:
            return False, already_detected

        ious = np.array([self._calculate_iou(prediction_rect, g) for g in ground_truth[image_id]])
        best_bb = np.argmax(ious)
        best_iou = ious[best_bb]

        if best_iou < iou_threshold or (image_id, best_bb) in already_detected:
            return False, already_detected

        already_detected.add((image_id, best_bb))
        return True, already_detected

    def _evaluate_predictions(self, ground_truths, predictions, iou_threshold):
        """ Evaluate the correctness of the given predictions.
        Args:
            ground_truths: List of ground truths for the class. {image_id: [[left, top, right, bottom], [...]], ...}
            predictions: List of predictions for the class. [[image_id, probability, left, top, right, bottom], [...], ...]
            iou_threshold: Minimum IOU hreshold to be considered as a same bounding box.
        """

        # Sort the predictions by the probability
        sorted_predictions = sorted(predictions, key=lambda x: -x[1])
        already_detected = set()
        is_correct = []
        for prediction in sorted_predictions:
            correct, already_detected = self._is_true_positive(prediction, ground_truths, already_detected,
                                                               iou_threshold)
            is_correct.append(correct)

        is_correct = np.array(is_correct)
        probabilities = np.array([p[1] for p in sorted_predictions])

        return is_correct, probabilities

    def _calculate_average_precision(self, is_correct, probabilities, true_num):
        if true_num == 0:
            return 0
        if not is_correct or not any(is_correct):
            return 0
        recall = float(np.sum(is_correct)) / true_num
        return sklearn.metrics.average_precision_score(is_correct, probabilities) * recall

    def get_report(self):
        all_aps = []
        for class_index in self.is_correct:
            ap = self._calculate_average_precision(self.is_correct[class_index],
                                                   self.probabilities[class_index],
                                                   self.true_num[class_index])
            all_aps.append(ap)

        mean_ap = statistics.mean(all_aps) if all_aps else 0
        return {'mAP_{}'.format(int(self.iou*100)): mean_ap}

    def reset(self):
        self.is_correct = collections.defaultdict(list)
        self.probabilities = collections.defaultdict(list)
        self.true_num = collections.defaultdict(int)


class ObjectDetectionEvaluator(Evaluator):
    def __init__(self, iou_values=[0.3, 0.5, 0.75, 0.9]):
        self.evaluators = [ObjectDetectionSingleIOUEvaluator(iou) for iou in iou_values]
        super(ObjectDetectionEvaluator, self).__init__()

    def add_predictions(self, predictions, targets):
        for evaluator in self.evaluators:
            evaluator.add_predictions(predictions, targets)

    def get_report(self):
        report = {}
        for evaluator in self.evaluators:
            report.update(evaluator.get_report())
        return report

    def reset(self):
        for evaluator in self.evaluators:
            evaluator.reset()
