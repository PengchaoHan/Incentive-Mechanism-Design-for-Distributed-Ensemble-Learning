import numpy as np
from config import device

def draw_tsne(dataloader, classifier, save_img_dir, n_feature, n_components=2):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    F = []
    Y = []
    for i, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        features = classifier.model(images).view(-1, n_feature)
        predictions = features.data.max(1)[1]
        indices = predictions.eq(labels.data.view_as(predictions))
        F.extend(features[indices].tolist())
        Y.extend(labels[indices].tolist())
    F = np.array(F)#[0:5000]
    Y = np.array(Y)#[0:5000]

    tsne = TSNE(n_components=n_components, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(F)

    # from sklearn import decomposition
    # fa = decomposition.FactorAnalysis(n_components=n_components)
    # X_tsne = fa.fit_transform(X)

    # from sklearn import decomposition
    # pca = decomposition.PCA(n_components=n_components)
    # X_tsne = pca.fit_transform(X)

    # from sklearn import decomposition
    # fica = decomposition.FastICA(n_components=n_components)
    # X_tsne = fica.fit_transform(X)

    # from sklearn import manifold
    # from sklearn.metrics import euclidean_distances
    # similarities = euclidean_distances(X)
    # mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, dissimilarity="precomputed", n_jobs=1)
    # X_tsne = mds.fit(similarities).embedding_

    if n_components == 2:
        # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=Y)
        # plt.colorbar()
        plot_embedding(X_tsne, Y, "t-SNE 2D")
    if n_components == 3:
        plot_embedding_3d(X_tsne, Y, "t-SNE 3D ")
    # plt.show()
    fig = plt.gcf()
    fig.savefig(save_img_dir + '_tsne.jpg', dpi=100)
    fig.savefig(save_img_dir + '_tsne.pdf')


def plot_embedding_3d(X, Y, title=None):
    color = ['darkred', 'darkblue', 'darkgreen', 'chocolate', 'gold', 'hotpink', 'dimgray', 'blueviolet', 'steelblue',
             'darkorange']
    import matplotlib.pyplot as plt
    # to [0,1]
    x_min, x_max = np.min(X,axis=0), np.max(X,axis=0)
    X = (X - x_min) / (x_max - x_min)
    fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax = plt.gca(projection='3d')
    for i in range(X.shape[0]):
        if Y[i] < 8:
            col = plt.cm.Set1(Y[i])
        else:
            if Y[i] < 15:
                col = plt.cm.Set2(Y[i] - 8)
            else:
                col = plt.cm.Set3(Y[i] - 15)
        # ax.text(X[i, 0], X[i, 1], X[i,2],str(Y[i]),
        #          color=col,
        #          fontdict={'weight': 'bold', 'size': 9})
        ax.plot(X[i, 0], X[i, 1], X[i,2],
                 color=col,
                 fontdict={'weight': 'bold', 'size': 9})
    if title is not None:
        plt.title(title)


def plot_embedding(data, label, title):
    import matplotlib.pyplot as plt
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    # ax = plt.subplot(111)
    for i in range(data.shape[0]):
        if label[i] < 8:
            col = plt.cm.Set1(label[i])
        else:
            if label[i] < 15:
                col = plt.cm.Set2(label[i] - 8)
            else:
                col = plt.cm.Set3(label[i] - 15)
        # plt.text(data[i, 0], data[i, 1], str(label[i]),
        #          color=col,
        #          fontdict={'weight': 'bold', 'size': 9})
        plt.plot(data[i, 0], data[i, 1],
                 color=col, marker='o', markersize=1)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig
