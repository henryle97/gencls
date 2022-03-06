
from .metric import accuracy,  precision_recall_f1, calculate_confusion_matrix

def build_metric(config):
    metric_value = config['Metric']
    if  metric_value == 'accuracy':
        return accuracy
    elif metric_value == 'f1':
        return precision_recall_f1
    elif metric_value == 'confusion':
        return calculate_confusion_matrix
    else:
        raise NotImplementedError(f"Not implement metric {metric_value}")