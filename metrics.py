import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix, classification_report

@st.cache(allow_output_mutation=True)
def load_data(file):
    df = pd.read_csv(file)
    return df

@st.cache
def plot_roc_curve(y_true, y_pred, label):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC = %0.2f)' % roc_auc)
    plt.legend(loc="lower right")
    return roc_auc

@st.cache(allow_output_mutation=True)
def interactive_scatter_plot(df):
    df['error'] = abs(df['predicted'] - df['actual'])
    fig = px.scatter(df, x='predicted', y='actual', color='error', color_continuous_scale='Viridis')
    fig.update_layout(title='Predicted vs Actual (Colored by Absolute Error)', xaxis_title='Predicted', yaxis_title='Actual')
    
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

        y_pred_bin = [1 if p >= threshold else 0 for p in y_pred]
        y_true_bin = [1 if p >= threshold else 0 for p in y_true]

        st.write("Confusion Matrix:")
        cm = confusion_matrix(y_true_bin, y_pred_bin)
        st.write(cm)

        st.write("Classification Report:")
        cr = classification_report(y_true_bin, y_pred_bin, output_dict=True)
        cr = pd.DataFrame(cr)
        st.dataframe(cr) 

        st.write("ROC Curve:")
        plt.figure(figsize=(8,6))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plot_roc_curve(y_true_bin, y_pred_bin, f"Threshold = {threshold}")
        st.pyplot()
        
        st.write("AUC:")
        auc = roc_auc_score(y_true_bin, y_pred_bin)
        st.write(auc)

if __name__ == "__main__":
    main()
