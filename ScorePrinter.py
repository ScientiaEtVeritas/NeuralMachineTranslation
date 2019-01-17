class ScorePrinter:
    def __init__(self, prefix, metrics):
        self.metrics = metrics
        self.scores = {metric[0]:[] for metric in metrics}
        self.count = 0
        self.prefix = prefix
        
    def update(self, input, output, ground_truth, nll):
        self.count += 1
        for (metric_name, metric_func) in self.metrics:
            self.scores[metric_name].append(metric_func(input, output, ground_truth, nll))
            
    def printAvg(self, showCount = True, last = None):
        last = last or self.count
        print(f" \n[{self.prefix}] {str(self.count) + ' examples /' if showCount else ''} ", end = ' ')
        for (metric_name, _) in self.metrics:
            avg_score = sum(self.scores[metric_name][-last:]) / last
            print("{0}: {1:.2f}".format(metric_name, avg_score), end=' ')
            
    def startEpoch(self, epoch):
        print(f"\n\nEpoch {epoch} started")
              
    def endEpoch(self, epoch):
        print(f"\n\nEpoch {epoch} ended")