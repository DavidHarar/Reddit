from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import datetime


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def report(y_true, y_pred,y_score):
  fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
  roc_auc = np.round(metrics.auc(fpr, tpr),4)
  precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_score)
  pr_auc = np.round(metrics.auc(recall, precision),4)
  bacc = np.round(metrics.balanced_accuracy_score(y_true, y_pred),4)

  # plots
  plot_df = pd.DataFrame(
      {
          'Label':y_true,
          'Prediction':y_score,
          'Predicted Label':y_pred
          }
      )

  fig, axs = plt.subplots(1, 4, figsize = (12,3))
  sns.histplot(data = plot_df, x = 'Prediction', hue = 'Label', ax=axs[0])
  axs[0].set_title('Scores Distribution')

  axs[1].plot(fpr, tpr)
  axs[1].set_xlabel("False Positive Rate")
  axs[1].set_ylabel("True Positive Rate")
  axs[1].set_title('ROC')

  axs[2].plot(recall, precision)
  axs[2].set_xlabel("Recall")
  axs[2].set_ylabel("Precision")
  axs[2].set_title('PR Curve')

  crosstab = pd.crosstab(plot_df['Label'], plot_df['Predicted Label'], normalize='index')
  sns.heatmap(crosstab, annot=True, fmt='.2f', ax = axs[3])
  plt.tight_layout()
  plt.show()

  # prints
  print(f"Balanced Accuracy: {bacc}")
  print(f"AUC AUC:           {roc_auc}")
  print(f"PR AUC:            {pr_auc}")

# %%
