import numpy as np


class ScorePrinter:
    def __init__(self, prefix, metrics):
        self.metrics = metrics
        self.prefix = prefix

    def update(self, *args, **kwargs):
        self.count += 1
        for (metric_name, metric_func) in self.metrics:
            self.scores[metric_name].append(metric_func(**kwargs))

    def printAvg(self, showCount=True, last=None):
        last = last or self.count
        print(
            f" \n[{self.prefix}] {str(self.count) + ' examples /' if showCount else ''} ",
            end=' ')

        avg_scores = self.getAvgScores(last)

        for metric_name in avg_scores.keys():
            avg_score = avg_scores[metric_name]
            if not isinstance(avg_score, list):
                print("{0}: {1:.2f}".format(metric_name, avg_score), end=' ')
            else:
                print("\t".join(["{0}[{1}]: {2:.2f}".format(
                    metric_name, i, float(x)) for i, x in enumerate(avg_score)]))

    def startEpoch(self, epoch):
        print(f"\n\nEpoch {epoch} started")
        self.beginMeasurements()

    def beginMeasurements(self):
        self.scores = {metric_name: [] for metric_name, _ in self.metrics}
        self.count = 0

    def endEpoch(self, epoch):
        print(f"\n\nEpoch {epoch} ended")

    def getAvgScores(self, last=None):
        last = last or self.count
        avg_scores = dict()
        for (metric_name, _) in self.metrics:
            examples = self.scores[metric_name][-last:]
            avg_score = sum(examples) / len(examples)
            if isinstance(avg_score, np.ndarray):
                avg_scores[metric_name] = avg_score.tolist()
            else:
                avg_scores[metric_name] = avg_score
        return avg_scores
