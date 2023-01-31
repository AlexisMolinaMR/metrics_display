import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix, classification_report

@st.cache(allow_output_mutation=True)
def load_data(file):
    df = pd.read_csv(file)
    return df


@st.cache(allow_output_mutation=True)
def interactive_scatter_plot(df):
    df['error'] = abs(df['predicted'] - df['actual'])
    fig = px.scatter(df, x='predicted', y='actual', color='error', color_continuous_scale='Viridis')
    fig.update_layout(title='Predicted vs Actual (Colored by Absolute Error)', xaxis_title='Predicted', yaxis_title='Actual')
    
    return fig

@st.cache(allow_output_mutation=True)
def plot_roc_curve(y_true, y_pred, threshold):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC curve (AUC = {:.2f})'.format(auc)))
    fig.add_shape(type='line', x0=0, y0=0, x1=1, y1=1, xref='x', yref='y', line=dict(color='red', dash='dash'))
    fig.update_layout(title='Receiver Operating Characteristic (ROC) Curve - {}'.format(threshold), xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
    
    return fig


def main():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title("ROC Curve and AUC Analysis")
    file = st.file_uploader("Upload your csv file", type=["csv"])
    
    if file is not None:
        df = load_data(file)
        y_true = df["actual"].values
        y_pred = df["predicted"].values

        fig = interactive_scatter_plot(df)
        st.plotly_chart(fig)

        threshold = st.slider("Threshold",-13.,10., -5.5, 0.1)

        y_pred_bin = ['bad' if p >= threshold else 'good' for p in y_pred]
        y_true_bin = ['bad' if p >= threshold else 'good' for p in y_true]

        st.write("Confusion Matrix:")
        cm = confusion_matrix(y_true_bin, y_pred_bin)
        
        ax= plt.subplot()
        sns.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

        # labels, title and ticks
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        ax.xaxis.set_ticklabels(['bad', 'good'])
        ax.yaxis.set_ticklabels(['bad', 'good'])
        st.pyplot()

        st.write("Classification Report:")
        cr = classification_report(y_true_bin, y_pred_bin, output_dict=True)
        cr = pd.DataFrame(cr)
        st.dataframe(cr) 

        y_pred_bin = [0 if p >= threshold else 1 for p in y_pred]
        y_true_bin = [0 if p >= threshold else 1 for p in y_true]

        st.write("ROC Curve:")
       # plt.figure(figsize=(8,6))
       # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
       # plt.xlim([0.0, 1.0])
       # plt.ylim([0.0, 1.0])
       # plt.xlabel('False Positive Rate')
       # plt.ylabel('True Positive Rate')
     #   plot_roc_curve(y_true_bin, y_pred_bin, f"Threshold = {threshold}")
        fig1 = plot_roc_curve(y_true_bin, y_pred_bin, f"Threshold = {threshold}")
        st.plotly_chart(fig1)
        
    #    st.write("AUC:")
    #    auc = roc_auc_score(y_true_bin, y_pred_bin)
    #    st.write(auc)

if __name__ == "__main__":
    main()
