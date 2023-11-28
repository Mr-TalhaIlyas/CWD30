import torch

from data.utils import collate, images_transform

import numpy as np
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#%%
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def values_fromreport(report):
    p = report['weighted avg']['precision']
    r = report['weighted avg']['recall']
    f1 = report['weighted avg']['f1-score']
    return p,r, f1

def get_model_embeddings(model, dataloader, layer_name):
    embeddings = []
    labels = []
    pbar = tqdm(dataloader, total=len(dataloader))
    # Create a hook to extract the output of the specified layer
    def hook(module, input, output):
        embeddings.append(output.cpu().numpy())

    model.eval() # set model to evaluation mode
    with torch.no_grad():
        # Register the hook for the specified layer
        hook_handle = getattr(model, layer_name).register_forward_hook(hook)
        # hook_handle = model.classifier[0].register_forward_hook(hook)

        for step, data_batch in enumerate(pbar):
            # prepare data
            image = images_transform(data_batch['img'])
            output = model(image)
            label = torch.from_numpy(np.asarray(data_batch['lbl'])).to(DEVICE)

            labels.append(label.cpu().numpy())

            pbar.set_description(f'Extracting Features')
        # Remove the hook
        hook_handle.remove()
    if embeddings and labels:
            return np.vstack(embeddings), np.hstack(labels)
    else:
        raise ValueError("No embeddings or labels found. Please check your model and dataloader.")
    


def plot_tsne(embeddings, labels, legends=False):
    # plt.figure(figsize=(10, 10))
    classes = np.unique(labels)
    sns.set(style="whitegrid")
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels,  cmap='Spectral' , s=1)
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(len(classes)))#.set_ticks(np.arange(len(classes)))
    if legends:
        plt.legend(*scatter.legend_elements(), title='Classes')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
    plt.show()

def plot_tsne_3d(embeddings, labels):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], c=labels, cmap='Spectral', s=1)
    plt.legend(*scatter.legend_elements(), title='Classes')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_zlabel('t-SNE 3')
    plt.show()


# def plot_tsne(embeddings, labels, text_labels=None):
#     """
#     Plot the t-SNE plot with optional text legends in a grid format below the plot.

#     :param embeddings: t-SNE embeddings (2D array-like)
#     :param labels: Labels for each data point (1D array-like)
#     :param text_labels: Optional list of text labels to be used as legends (default: None)
#     """
#     unique_labels = np.unique(labels)
#     colors = plt.cm.get_cmap('viridis', len(unique_labels))

#     # Plot the t-SNE points
#     for label in unique_labels:
#         indices = np.where(labels == label)
#         legend_label = text_labels[int(label)] if text_labels else label
#         plt.scatter(embeddings[indices, 0], embeddings[indices, 1], c=[colors(label / len(unique_labels))], label=legend_label)

#     plt.xlabel("t-SNE Component 1")
#     plt.ylabel("t-SNE Component 2")

#     # Create a legend grid below the plot
#     ax = plt.gca()
#     ax_legend = inset_axes(ax, width="100%", height="5%", loc='lower center', borderpad=0)
#     ax_legend.set_axis_off()

#     ncol = 6  # Number of columns in the legend grid
#     markers = [plt.Line2D([0, 0], [0, 1], color=colors(label / len(unique_labels)), linestyle='', marker='o') for label in unique_labels]
#     ax_legend.legend(markers, text_labels, ncol=ncol, mode='expand', borderaxespad=0, frameon=False)

#     plt.show()