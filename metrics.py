import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report, r2_score

@st.cache(allow_output_mutation=True)
def load_data(file):
    df = pd.read_csv(file)
    return df

@st.cache(allow_output_mutation=True)
def interactive_scatter_plot(df):
    df['error'] = abs(df['predicted'] - df['actual'])
    fig = px.scatter(df, x='predicted', y='actual', color='error', color_continuous_scale='Viridis')
    fig.update_layout(title='Predicted vs Actual (Colored by Absolute Error)', xaxis_title='Predicted', yaxis_title='Actual')
    
    # Add R² computation
    fig.update_traces(mode='markers',
                      marker=dict(size=12,
                                  line=dict(width=2,
                                            color='DarkSlateGrey')))
    
    fig.update_layout(updatemenus=[
        dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(
                    label="Show R²",
                    method="update",
                    args=[
                        {"visible": [True, False]},
                        {
                            "title": "Predicted vs Actual (Colored by Absolute Error) - R² = {:.2f}".format(compute_r2(df, fig.data[0].selectedpointssrc)),
                        }
                    ]
                ),
                dict(
                    label="Hide R²",
                    method="update",
                    args=[
                        {"visible": [False, True]},
                        {
                            "title": "Predicted vs Actual (Colored by Absolute Error)",
                        }
                    ]
                ),
            ]
        )
    ])

    return fig

def compute_r2(df, selected_points_src):
    selected_points = pd.DataFrame({"predicted": [df.loc[i]['predicted'] for i in selected_points_src['points']],
                                    "actual": [df.loc[i]['actual'] for i in selected_points_src['points']]})
    r2 = r2_score(selected_points['actual'], selected_points['predicted'])
    
    return r2

#@st.cache(allow_output_mutation=True)
#def interactive_scatter_plot(df):
#    df['error'] = abs(df['predicted'] - df['actual'])
#    fig = px.scatter(df, x='predicted', y='actual', color='error', color_continuous_scale='Viridis')
#    fig.update_layout(title='Predicted vs Actual (Colored by Absolute Error)', xaxis_title='Predicted', yaxis_title='Actual')
    
#    return fig

@st.cache(allow_output_mutation=True)
def plot_roc_curve(y_true, y_pred, threshold_actual, threshold_preds):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC curve (AUC = {:.2f})'.format(auc)))
    fig.add_shape(type='line', x0=0, y0=0, x1=1, y1=1, xref='x', yref='y', line=dict(color='red', dash='dash'))
    fig.update_layout(title='ROC Curve - Actual threshold: {} - Predicted threshold: {} - AUC: {:.2f}'.format(threshold_actual, threshold_preds, auc), xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
    
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

        threshold_actual = st.slider("Threshold on actual values",-13.,10., -5.5, 0.1)
        threshold_preds = st.slider("Threshold on predicted values",-13.,10., -5.5, 0.1)

        y_pred_bin = ['bad' if p >= threshold_preds else 'good' for p in y_pred]
        y_true_bin = ['bad' if p >= threshold_actual else 'good' for p in y_true]

        st.write("Confusion Matrix:")
        cm = confusion_matrix(y_true_bin, y_pred_bin)
        
        ax= plt.subplot()
        sns.heatmap(cm, annot=True, fmt='g', ax=ax)

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

        y_pred_bin = [0 if p >= threshold_preds else 1 for p in y_pred]
        y_true_bin = [0 if p >= threshold_actual else 1 for p in y_true]

        st.write("ROC Curve:")
      
        try:
            fig1 = plot_roc_curve(y_true_bin, y_pred_bin, threshold_preds, threshold_actual)
            st.plotly_chart(fig1)
        except:
            st.error("Error. One of the classes run out of samples.")
  

if __name__ == "__main__":
    main()
