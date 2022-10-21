from sklearn import metrics
from pytorch_utils import forward
import numpy as np
import warnings
warnings.filterwarnings("ignore")


class Evaluator(object):
    def __init__(self, model):
        """Evaluator.

        Args:
          model: object
        """
        self.model = model
        
    def evaluate(self, data_loader):
        """Forward evaluation data and calculate statistics.

        Args:
          data_loader: object

        Returns:
          statistics: dict, 
              {'average_precision': (classes_num,), 'auc': (classes_num,)}
        """
        # Forward
        output_dict = forward(
            model=self.model, 
            generator=data_loader, 
            return_target=True)

        clipwise_output = output_dict['clipwise_output']    # (audios_num, classes_num)
        target = output_dict['target']    # (audios_num, classes_num)

        average_precision = metrics.average_precision_score(
            target, clipwise_output, average=None)

        roc_auc = metrics.roc_auc_score(target, clipwise_output, average=None)

        # convert the probabilities to class labels
        predicted_label=np.argmax(clipwise_output, axis=1)
        true_label=np.argmax(target, axis=1)
      
        F1_score=metrics.f1_score(true_label, predicted_label, average=None)
        acc_score=metrics.accuracy_score(true_label,predicted_label)
        rall_score=metrics.recall_score(true_label,predicted_label,average=None)
        
        statistics = {'average_precision': average_precision, 'roc_auc': roc_auc, 'F1_score':F1_score, 'acc_score':acc_score,"recall_score":rall_score}

        return statistics