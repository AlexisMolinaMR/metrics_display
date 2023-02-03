import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
import torch
from torch import Tensor
import networkx as nx
import umap
from karateclub.graph_embedding import Graph2Vec
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

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
def plot_roc_curve(y_true, y_pred, threshold_actual, threshold_preds):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC curve (AUC = {:.2f})'.format(auc)))
    fig.add_shape(type='line', x0=0, y0=0, x1=1, y1=1, xref='x', yref='y', line=dict(color='red', dash='dash'))
    fig.update_layout(title='ROC Curve - Actual threshold: {} - Predicted threshold: {} - AUC: {:.2f}'.format(threshold_actual, threshold_preds, auc), xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
    
    return fig

x_map = {
    "atomic_num": list(range(0, 119)),
    "chirality": [
        "CHI_UNSPECIFIED",
        "CHI_TETRAHEDRAL_CW",
        "CHI_TETRAHEDRAL_CCW",
        "CHI_OTHER",
    ],
    "degree": list(range(0, 11)),
    "formal_charge": list(range(-5, 7)),
    "num_hs": list(range(0, 9)),
    "num_radical_electrons": list(range(0, 5)),
    "hybridization": [
        "UNSPECIFIED",
        "S",
        "SP",
        "SP2",
        "SP3",
        "SP3D",
        "SP3D2",
        "OTHER",
    ],
    "is_aromatic": [False, True],
    "is_in_ring": [False, True],
}

e_map = {
    "bond_type": [
        "misc",
        "SINGLE",
        "DOUBLE",
        "TRIPLE",
        "AROMATIC",
    ],
    "stereo": [
        "STEREONONE",
        "STEREOZ",
        "STEREOE",
        "STEREOCIS",
        "STEREOTRANS",
        "STEREOANY",
    ],
    "is_conjugated": [False, True],
}

def from_smiles(smiles: str, with_hydrogen: bool = False, kekulize: bool = False):
    r"""Converts a SMILES string to a :class:`torch_geometric.data.Data`
    instance.

    Args:
        smiles (string, optional): The SMILES string.
        with_hydrogen (bool, optional): If set to :obj:`True`, will store
            hydrogens in the molecule graph. (default: :obj:`False`)
        kekulize (bool, optional): If set to :obj:`True`, converts aromatic
            bonds to single/double bonds. (default: :obj:`False`)
    """
    from rdkit import Chem, RDLogger

    from torch_geometric.data import Data

    RDLogger.DisableLog("rdApp.*")

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        mol = Chem.MolFromSmiles("")
    if with_hydrogen:
        mol = Chem.AddHs(mol)
    if kekulize:
        mol = Chem.Kekulize(mol)

    xs = []
    for atom in mol.GetAtoms():
        x = []
        x.append(x_map["atomic_num"].index(atom.GetAtomicNum()))
        x.append(x_map["chirality"].index(str(atom.GetChiralTag())))
        x.append(x_map["degree"].index(atom.GetTotalDegree()))
        x.append(x_map["formal_charge"].index(atom.GetFormalCharge()))
        x.append(x_map["num_hs"].index(atom.GetTotalNumHs()))
        x.append(x_map["num_radical_electrons"].index(atom.GetNumRadicalElectrons()))
        x.append(x_map["hybridization"].index(str(atom.GetHybridization())))
        x.append(x_map["is_aromatic"].index(atom.GetIsAromatic()))
        x.append(x_map["is_in_ring"].index(atom.IsInRing()))
        xs.append(x)

    x = torch.tensor(xs, dtype=torch.long).view(-1, 9)

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        e = []
        e.append(e_map["bond_type"].index(str(bond.GetBondType())))
        e.append(e_map["stereo"].index(str(bond.GetStereo())))
        e.append(e_map["is_conjugated"].index(bond.GetIsConjugated()))

        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]

    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 3)

    if edge_index.numel() > 0:  # Sort indices.
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)
    
def to_networkx(data):
    '''
    Convert a torch_geometric.data.Data object to a networkx.Graph object
    '''
    G = nx.Graph()
    for i in range(data.x.size(0)):
        G.add_node(i)
    for i, j in data.edge_index.t().tolist():
        G.add_edge(i, j)
    return G


def main():
    st.set_option('deprecation.showPyplotGlobalUse', False) 
    
    st.title("ROC Curve and AUC Analysis")
    file = st.file_uploader("Upload your csv file", type=["csv"])
    
    if file is not None:
        df = load_data(file)
        y_true = df["actual"].values
        y_pred = df["predicted"].values

        div_ana = st.checkbox('Perform diversity analysis. Note that this process may take a while.')

        if div_ana:
            
            if 'smiles' in df.columns:

                st.title("Diversity Analysis")
        
                smiles = list(df["smiles"])
                dockscores = list(df["actual"])
                classes = ['prediction' for i in range(len(smiles))]

                file_train = st.file_uploader("Upload your csv file with training compounds", type=["csv"])

                if file_train is not None:
                    df_train = load_data(file_train)

                    smiles_train = list(df_train["smiles"])
                    classes = ['training' for i in range(len(smiles_train))] + ['prediciton' for i in range(len(smiles))]
                    smiles = smiles_train + smiles 
                
                    dockscores = list(df_train["dockscore"]) + list(df["actual"])

                    graphs = []

                    my_bar = st.progress(0)
                    smiles_counter = 0

                    for smile in smiles:
                        data = from_smiles(smile)
                        G = to_networkx(data)
                        graphs.append(G)
                        smiles_counter += 1
                        my_bar.progress(((smiles_counter)/len(smiles))) 

                    st.info('Creating graph2vec embedding...')

                    model = Graph2Vec()
                    model.fit(graphs)
                    emb = model.get_embedding()

                
                else:

                    graphs = []

                    my_bar = st.progress(0)
                    smiles_counter = 0

                    for smile in smiles:
                        data = from_smiles(smile)
                        G = to_networkx(data)
                        graphs.append(G)
                        smiles_counter += 1
                        my_bar.progress(((smiles_counter)/len(smiles))) 


                    st.info('Creating graph2vec embedding...')

                    model = Graph2Vec()
                    model.fit(graphs)
                    emb = model.get_embedding()

                st.info('Fitting projection...')

                fit = umap.UMAP()
                u = fit.fit_transform(emb)

                st.success('Done!')

                fig_div1 = px.scatter(x=u[:, 0], y=u[:, 1], color=classes)
                st.plotly_chart(fig_div1)

                dataviz = pd.DataFrame(u)
                dataviz.columns = ['x', 'y']

                dataviz['smiles'] = smiles
                dataviz['dockscore'] = dockscores

                fig_div2 = px.scatter(dataviz, x='x', y='y', color='dockscore', hover_data=['smiles'])
                fig_div2.update_traces(
                    hovertext=smiles
                    )

                st.plotly_chart(fig_div2)

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
