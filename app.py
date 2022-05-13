import streamlit as st
import numpy as np
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score, roc_curve, matthews_corrcoef
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

models = {
    "LSTM": keras.models.load_model('models/LSTM.h5'),
    "S-LSTM": keras.models.load_model('models/S-LSTM.h5'),
    "B-LSTM": keras.models.load_model('models/B-LSTM.h5'),
    "CNN-LSTM": keras.models.load_model('models/CNN-LSTM.h5'),
    "CNN-BLSTM": keras.models.load_model('models/CNN-BLSTM.h5')
}

def make_predict_result(data, model_name): 
    features = data.columns[0: -1]
    X_test, y_test =  np.array(data.iloc[:, :-1]), np.array(data.iloc[:, -1])
    model = models[model_name]
    pred = (model.predict(X_test) > 0.5) * 1 
    fpr, tpr, _ = roc_curve(y_test, pred)
    result = {
        "Model": model_name,
        "confusion_matrix": confusion_matrix(y_test, pred),
        "y_test": y_test,
        "pred": pred,
        "ACC": model.evaluate(X_test, y_test, verbose=0)[1],
        "MCC": matthews_corrcoef(y_test, pred),
        "auc": roc_auc_score(y_test, pred),
        "fpr": fpr,
        "tpr": tpr
    }
    return result

def make_roc_figure(results):
    fig, ax = plt.subplots(figsize=(10, 8))
    if len(results) == 0:
        return fig 
    
    result_table = pd.DataFrame(columns=['Model', 'fpr','tpr','auc'])
    for res in results:
        result_table = result_table.append({
            'Model': res["Model"],
            'fpr': res["fpr"],
            'tpr': res["tpr"],
            'auc': res["auc"]
        }, ignore_index=True)

    result_table.set_index('Model', inplace=True)
    for model in result_table.index:
        result = result_table.loc[model]
        ax.plot(
            result['fpr'],
            result['tpr'],
            label="{}, AUC={:.3f}".format(model, result['auc'])
        )
    
    ax.plot([0,1], [0,1], color='orange', linestyle='--')
    ax.set_xticks(np.arange(0.0, 1.1, step=0.1))
    ax.set_xlabel("Flase Positive Rate", fontsize=15)
    ax.set_yticks(np.arange(0.0, 1.1, step=0.1))
    ax.set_ylabel("True Positive Rate", fontsize=15)
    ax.set_title('ROC Curve Analysis', fontweight='bold', fontsize=15)
    ax.legend(prop={'size':12}, loc='lower right')
    return fig 

def make_cm_plot(result): 
    fig, ax = plt.subplots(figsize=(10, 8))
    name = result["Model"]
    ax.set_title(f'confusion matrix of {name}', fontweight='bold', fontsize=15)
    cmp = ConfusionMatrixDisplay(result["confusion_matrix"], display_labels=[0, 1])
    cmp.plot(ax=ax)
    return fig

st.header("Protein Glutarylation Sites Prediction")
data = pd.read_csv('./dataset/Test_Encoded.csv', index_col=0) 
data = shuffle(data)

uploaded_file = st.file_uploader("Choose a CSV file")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

models_name = models.keys()
options = st.multiselect('Choose models', models_name, models.keys())

results = [ make_predict_result(data, opt) for opt in options ]
fig = make_roc_figure(results)
st.pyplot(fig)

cmps = [ make_cm_plot(result) for result in results ]
if len(cmps) != 0:
    for cmp in cmps:
        st.pyplot(cmp)
