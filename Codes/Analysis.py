import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def roc_curve_for_analysis(figsize: tuple, roc_data: dict, title: str, save_name: str, save: bool=False) -> None:
    #Set figure size
    plt.figure(figsize = figsize)
    #For every key and their corresponding data, create the roc_curve
    for model_name, data in roc_data.items():
        fpr, tpr, _ = roc_curve(data["labels"], data["scores"], pos_label=1)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")  # Diagonal line
    plt.xlabel("False Positive Rate")
    plt.xlim([0,1])
    plt.ylim([0,1.1])
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.grid()
    if save:
        plt.savefig(save_name)
    plt.show()