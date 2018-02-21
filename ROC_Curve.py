import numpy as np
from sklearn import *
import matplotlib.pyplot as plt
import os

def calculate_eer_auc_ap(label,distance):

    fpr, tpr, thresholds = metrics.roc_curve(label, distance, pos_label=1)
    AUC = metrics.roc_auc_score(label, distance, average='macro', sample_weight=None)
    AP = metrics.average_precision_score(label, distance, average='macro', sample_weight=None)

    # Calculating EER
    intersect_x = fpr[np.abs(fpr - (1 - tpr)).argmin(0)]
    EER = intersect_x

    return EER,AUC,AP,fpr, tpr


def Plot_ROC_Fn(label,distance):

    fpr, tpr, thresholds = metrics.roc_curve(label, distance, pos_label=1)
    AUC = metrics.roc_auc_score(label, distance, average='macro', sample_weight=None)
    # AP = metrics.average_precision_score(label, -distance, average='macro', sample_weight=None)

    # Calculating EER
    intersect_x = fpr[np.abs(fpr - (1 - tpr)).argmin(0)]
    EER = intersect_x
    print("EER = ", float(("{0:.%ie}" % 1).format(intersect_x)))

    # AUC(area under the curve) calculation
    print("AUC = ", float(("{0:.%ie}" % 1).format(AUC)))

 #    EER =  0.0
	# AUC =  1.0

    # # AP(average precision) calculation.
    # # This score corresponds to the area under the precision-recall curve.
    # print("AP = ", float(("{0:.%ie}" % 1).format(AP)))

    # Plot the ROC
    fig = plt.figure()
    ax = fig.gca()
    lines = plt.plot(fpr, tpr, label='ROC Curve')
    plt.setp(lines, linewidth=2, color='r')
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    plt.title('ROC.jpg')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    # # Cutting the floating number
    # AUC = '%.2f' % AUC
    # EER = '%.2f' % EER
    # # AP = '%.2f' % AP
    #
    # # Setting text to plot
    # # plt.text(0.5, 0.6, 'AP = ' + str(AP), fontdict=None)
    # plt.text(0.5, 0.5, 'AUC = ' + str(AUC), fontdict=None)
    # plt.text(0.5, 0.4, 'EER = ' + str(EER), fontdict=None)
    plt.grid()
    plt.show()

def Plot_PR_Fn(label,distance):

    precision, recall, thresholds = metrics.precision_recall_curve(label, distance, pos_label=1, sample_weight=None)
    AP = metrics.average_precision_score(label, distance, average='macro', sample_weight=None)

    # AP(average precision) calculation.
    # This score corresponds to the area under the precision-recall curve.
    print("AP = ", float(("{0:.%ie}" % 1).format(AP)))

    # Plot the ROC
    fig = plt.figure()
    ax = fig.gca()
    lines = plt.plot(recall, precision, label='ROC Curve')
    plt.setp(lines, linewidth=2, color='r')
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    plt.title('PR.jpg')
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    # Cutting the floating number
    AP = '%.2f' % AP

    # Setting text to plot
    # plt.text(0.5, 0.5, 'AP = ' + str(AP), fontdict=None)
    plt.grid()
    plt.show()

def Plot_HIST_Fn(label,distance, num_bins = 50):

    dissimilarity = distance[:]
    gen_dissimilarity_original = []
    imp_dissimilarity_original = []
    for i in range(len(label)):
        if label[i] == 1:
            gen_dissimilarity_original.append(dissimilarity[i])
        else:
            imp_dissimilarity_original.append(dissimilarity[i])

    bins = np.linspace(np.amin(distance), np.amax(distance), num_bins)
    fig = plt.figure()
    plt.hist(gen_dissimilarity_original, bins, alpha=0.5, facecolor='blue', normed=False, label='gen_dist_original')
    plt.hist(imp_dissimilarity_original, bins, alpha=0.5, facecolor='red', normed=False, label='imp_dist_original')
    plt.legend(loc='upper right')
    plt.title('OriginalFeatures_Histogram.jpg')
    plt.show()

score = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),'score_vector.npy'))
label = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),'target_label_vector.npy'))


EER_temp,AUC_temp,AP,fpr, tpr = calculate_eer_auc_ap(label,score)
EER_VECTOR = EER_temp * 100
AUC_VECTOR = AUC_temp * 100

print(type(EER_VECTOR))
# <class 'numpy.float64'>
print(type(AUC_VECTOR))
# <class 'numpy.float64'>

print("EER=",np.mean(EER_VECTOR),np.std(EER_VECTOR))
print("AUC=",np.mean(AUC_VECTOR),np.std(AUC_VECTOR))

# EER= 0.0 0.0
# AUC= 100.0 0.0



Plot_ROC_Fn(label,score)
Plot_PR_Fn(label,score)
Plot_HIST_Fn(label,score)