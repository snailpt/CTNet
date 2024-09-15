import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd


def calculateConfusion(filename):
    data = pd.read_excel(filename, sheet_name=None)
    data_list = []
    for k, df in data.items():

        # Example true and predicted labels
        true_labels = df['true'].values
        pred_labels = df['pred'].values
        cm = confusion_matrix(true_labels, pred_labels)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        result = cm_normalized / cm_normalized.sum(axis=1, keepdims=True)
        data_list.append(result)
    return np.array(data_list)


def readPredTrue(filename):
    data = pd.read_excel(filename, sheet_name=None)
    data_list = []
    for k, v in data.items():
        data_list.append(v)
    
    df = pd.concat(data_list)
    
    # Example true and predicted labels
    true_labels = df['true'].values
    pred_labels = df['pred'].values
    
    # Generating the confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    result = cm_normalized / cm_normalized.sum(axis=1, keepdims=True)
    return result


cm_4_percentage = readPredTrue('pred_true_2a.xlsx')
categories_2a = ['Left', 'Right', 'Foot', 'Tongue'] 

cm_2_percentage = readPredTrue('pred_true_2b.xlsx')
categories_2b = ['Left', 'Right'] 



result_list = calculateConfusion('pred_true_2b.xlsx')
cm_2_percentage = result_list.mean(axis=0)


# Plotting
plt.figure(figsize=(12, 5))

# 4-Class Confusion Matrix
plt.subplot(1, 2, 1)
plt.imshow(cm_4_percentage, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Predicted labels', fontsize=20)
# plt.colorbar()
tick_marks = np.arange(len(['Left', 'Right', 'Foot', 'Tongue']))
plt.xticks(tick_marks, ['Left', 'Right', 'Foot', 'Tongue'], fontsize=15) # , rotation=45
plt.yticks(tick_marks, ['Left', 'Right', 'Foot', 'Tongue'], fontsize=15)
plt.xlabel('(a)', fontsize=25)
plt.ylabel('True labels', fontsize=20)
for i in range(len(['Left', 'Right', 'Foot', 'Tongue'])):
    for j in range(len(['Left', 'Right', 'Foot', 'Tongue'])):
        plt.text(j, i, "{:.2%}".format(cm_4_percentage[i, j]),
                 ha="center", va="center", color="white" if cm_4_percentage[i, j] > cm_4_percentage.max() / 2. else "black", fontsize=13)

# 2-Class Confusion Matrix
plt.subplot(1, 2, 2)
plt.imshow(cm_2_percentage, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Predicted labels', fontsize=20)
plt.colorbar()
tick_marks = np.arange(len(['Left', 'Right']))
plt.xticks(tick_marks, ['Left', 'Right'], fontsize=15) #, rotation=45
plt.yticks(tick_marks, ['Left', 'Right'],  fontsize=15)
plt.xlabel('(b)', fontsize=25)
plt.ylabel('True labels', fontsize=20)
for i in range(len(['Left', 'Right'])):
    for j in range(len(['Left', 'Right'])):
        plt.text(j, i, "{:.2%}".format(cm_2_percentage[i, j]),
                 ha="center", va="center", color="white" if cm_2_percentage[i, j] > cm_2_percentage.max() / 2. else "black", fontsize=13)

plt.tight_layout()
# plt.savefig('图3. confusion_matrices.png', dpi=300, bbox_inches='tight')  # 保存为png格式，设置dpi为600
plt.savefig('Fig4.pdf', dpi=300, bbox_inches='tight')  # 保存为png格式，设置dpi为600

plt.show()

