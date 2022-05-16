import streamlit as st
import numpy as np
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt
from utils import read_seq
from io import StringIO
from sklearn.utils import shuffle
from sklearn import preprocessing
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
    ax.set_title(f' {name} Confusion Matrix', fontweight='bold', fontsize=15)
    cmp = ConfusionMatrixDisplay(result["confusion_matrix"], display_labels=[0, 1])
    cmp.plot(ax=ax)
    return fig

st.header("DNN-E for Protein Glutarylation Sites Prediction")
data = pd.read_csv('./dataset/Test_Encoded.csv', index_col=0) 
data = shuffle(data)

with st.expander("Instructions"):
    st.write("""
        Deep Neural Network Framework Based on Word Embedding for Protein Glutarylation Sites Prediction
        Authors: Chuan-Ming Liu *, Van-Dai Ta, Nguyen Quoc Khanh Le, Direselign Addis Tadesse and Chongyang Shi *

        Instructions 
        This application is designed for testing the Deep Neural Network using the Embedding platform to predict the glutarylation sites.  The testing will be executed in the following steps
        
        First, load the input test sets under the FASTA format (.seq extension), including positive (glutarylation) and negative (non-glutarylation) sequences. As explained in the manuscript, 138 glutarylation sites (92 glutarylation, 46 non- glutarylation) has been used in this experiment. Both glutarylation and non- glutarylation sites will be concatenated into one single dataframe before encoded by label encoding. We only use the actual class of those sequences for evaluation, by plotting the confusion matrix, and ROC-AUC.
        
        After loading the input files, the model selection proposed to select the deep neural network model for glutarylation prediction. This allows use either to choose a specific model or multiple models to perform.
        The prediction results are automatically displayed as the model selected, including confusion matrix, ROC-AUC, and detailed results for each glutaration site. 
        For more detailed information about this project, please refer to our manuscript and Github repository
        https://github.com/AppliedComputingLab/BIO_DNNE

    """)

st.text('Load the test datasets(.sed file extension)')
pos_file = st.file_uploader("Choose a Positive seq file")
if pos_file is not None:
    file = StringIO(pos_file.getvalue().decode("utf-8"))
    pos_df, pos_dna = read_seq(file, isPositive=True)

neg_file = st.file_uploader("Choose a Negative seq file")
if neg_file is not None:
    file = StringIO(neg_file.getvalue().decode("utf-8"))
    neg_df, neg_dna = read_seq(file, isPositive=False)

label = preprocessing.LabelEncoder()
has_input =  (pos_file is not None) & (neg_file is not None)
if has_input:
    data = pd.concat([pos_df, neg_df], axis = 0)
    data = pd.concat([(data.iloc[:, :-1]).apply(label.fit_transform), data.iloc[:,-1:]], axis = 1)
    data_dna = pd.concat([pos_dna, neg_dna], axis = 0, ignore_index=True)

models_name = models.keys()
options = st.multiselect('Model selection', models_name, models.keys())

results = [ make_predict_result(data, opt) for opt in options ]
fig = make_roc_figure(results)
st.pyplot(fig)

cmps = [ make_cm_plot(result) for result in results ]
if len(cmps) != 0:
    for cmp in cmps:
        st.pyplot(cmp)


if has_input & len(results) != 0 :
    for result in results:
        name = result["Model"]
        df = pd.DataFrame(data = {
            "actual": result["y_test"],
            "predict": result["pred"].flatten()
        })
        df['validation'] = df.apply(lambda row: row.actual == row.predict, axis=1)
        df = pd.concat([data_dna, df], axis=1)
        st.header(name)
        st.dataframe(df)
