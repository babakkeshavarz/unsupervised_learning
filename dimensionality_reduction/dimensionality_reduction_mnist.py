import pandas as pd
import gzip
import pickle
import matplotlib.pyplot as plt
import seaborn as sns


def view_digit(example):
    label = y_train.loc[example]
    image = X_train.loc[example].values.reshape(28, 28)
    plt.title('Example: %d  Label: %d' % (example, label))
    plt.imshow(image.reshape(28, 28), cmap='gray')
    plt.show()

def scatterPlot(xDF, yDF, algoName):
    tempDF = pd.DataFrame(data=xDF.loc[:, 0:1], index=xDF.index)
    tempDF = pd.concat((tempDF, yDF), axis=1, join='inner')
    tempDF.columns = ['First Vector', 'Second Vector', 'Label']
    sns.lmplot(x='First Vector', y='Second Vector', data=tempDF, hue='Label', fit_reg=False)
    ax = plt.gca()
    ax.set_title(algoName)
    plt.show()


file_path = '..\\datasets_unsupervised_aapatel\mnist_data\\mnist.pkl.gz'

f = gzip.open(file_path , 'rb')

train_set, validation_set, test_set = pickle.load(f, encoding='latin1')

f.close()

X_train, y_train = train_set[0], train_set[1]
X_validation, y_validation = validation_set[0], validation_set[1]
X_test, y_test = test_set[0], test_set[1]

train_index = range(0, len(X_train))
validadtion_index = range(len(X_train), len(X_train) + len(X_validation))
test_index = range(len(X_train) + len(X_validation), len(X_train) + len(X_validation) + len(X_test))

X_train = pd.DataFrame(X_train , index=train_index)
y_train = pd.DataFrame(y_train , index=train_index)
# y_train = pd.DataFrame(y_train)



# view_digit(1)

############### Linear dimensionality reduction techniques ##################

from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import SparsePCA


n_components = 784
random_state = 42

pca = PCA(n_components=n_components, random_state=random_state)
incrementalPCA = IncrementalPCA(n_components=n_components, batch_size=None) ## when the data is too large
# sparsePCA = SparsePCA(n_components=n_components, alpha= 0.0001 , random_state=random_state , n_jobs=-1) ## when some degree of sparsity is required

X_train_pca = pca.fit_transform(X_train)
X_train_pca = pd.DataFrame(X_train_pca , index=X_train.index)

print("Variance of the first 10 dimensions: ", pca.explained_variance_ratio_[0:10])
print("Total information of PCA: ", sum(pca.explained_variance_ratio_))
print("Variance of the first 10 dimensions: ", sum(pca.explained_variance_ratio_[0:10]))


scatterPlot(X_train_pca, y_train, "Incremental PCA")



############### non-Linear dimensionality reduction techniques ##################
## Kernel PCA is a non-linear extension of PCA. It uses the kernel trick to project the data into a higher dimensional space and then apply PCA to extract the principal components.
from sklearn.decomposition import KernelPCA
n_components = 100
kernel = 'rbf'
gamma = None
random_state = 42
n_jobs = 4

KernelPCA = KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma, random_state=random_state, n_jobs=n_jobs)

X_train_kernelpca= KernelPCA.fit_transform(X_train.loc[0:2000 , :])
X_train_kernelpca = pd.DataFrame(X_train_kernelpca)

scatterPlot(X_train_kernelpca, y_train.loc[0:2000 , :], "Kernel PCA")


################ SVD ######################################
# SVD is more efficient since it foregoes the calculation of the covariance matrix
from sklearn.decomposition import TruncatedSVD

n_components = 200
algorithm = 'randomized'
n_iter = 5
random_state = 42

svd = TruncatedSVD(n_components=n_components, algorithm=algorithm, n_iter=n_iter, random_state=random_state)

X_train_svd = svd.fit_transform(X_train)
X_train_svd = pd.DataFrame(X_train_svd)

scatterPlot(X_train_svd, y_train, 'SVD')


##################### Random Projection  ############################
# In general PCA works well on relatively low dimensional data. Random Projection is a good alternative for high dimensional data.
## Random Projection is a computationally efficient way to reduce the dimensionality of the data by trading a controlled amount of accuracy (as additional variance) for faster processing times and smaller model sizes.
from sklearn.random_projection import GaussianRandomProjection

n_components = 'auto'
eps = 0.5
random_state = 42

GRP = GaussianRandomProjection(n_components=n_components, eps=eps, random_state=random_state)

X_train_GRP = GRP.fit_transform(X_train)
X_train_GRP = pd.DataFrame(X_train_GRP)

scatterPlot(X_train_GRP, y_train, 'GRP')



##################### Non-Linear Techniques ############################
##################### Isomap ###########################################
## Isomap is a non-linear dimensionality reduction method that finds a lower-dimensional embedding of the data that preserves the geodesic distances between all points.
## Isomap is computationally expensive
from sklearn.manifold import Isomap
n_neighbors = 5
n_components = 10
n_jobs = 4

isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components, n_jobs=n_jobs)

X_train_isomap = isomap.fit_transform(X_train.loc[0:2000 , :])

X_train_isomap = pd.DataFrame(X_train_isomap)

scatterPlot(X_train_isomap, y_train.loc[0:2000 , :], 'Isomap')

##################### Multidimensional Scaling (MDS) ############################
## takes a distance matrix as input and returns a matrix X that minimizes the difference between the Euclidean distances in the input and output spaces.
## MDS is used to visualize the dissimilarity between objects.
## computationally is very expensive
from sklearn.manifold import MDS
n_components = 2
n_init = 12
max_iter = 500
metric = True
n_jobs = 4
random_state = 42

mds = MDS(n_components=n_components, n_init=n_init, max_iter=max_iter, metric=metric, n_jobs=n_jobs, random_state=random_state)

# X_train_mds = mds.fit_transform(X_train.loc[0:1000 , :])
# X_train_mds = pd.DataFrame(X_train_mds)
# scatterPlot(X_train_mds, y_train.loc[0:1000 , :], 'MDS')

############## Locally Linear Embedding (LLE) ############################
## LLE is a non-linear dimensionality reduction technique that preserves the local distances between the samples.
## LLE is computationally expensive
from sklearn.manifold import LocallyLinearEmbedding
n_neighbors = 10
n_components = 2
method = 'standard'
random_state = 42
n_jobs = 4

lle = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components, method=method, random_state=random_state, n_jobs=n_jobs)

lle.fit(X_train.loc[0:1000 , :])  ## to speed up, fit only a subset of the data then transform the whole data
X_train_lle = lle.transform(X_train)
X_train_lle = pd.DataFrame(X_train_lle , index=train_index)

scatterPlot(X_train_lle, y_train, 'LLE')


##################### t-distributed Stochastic Neighbor Embedding (t-SNE) ############################
## t-SNE is a non-linear dimensionality reduction technique that is well-suited for embedding high-dimensional data for visualization in a low-dimensional space of two or three dimensions.
## t-SNE is used to visualize the similarity between objects.
## it is not used for feature engineering
## it's better to use PCA before t-SNE to reduce the number of dimensions and speed up the computation
## t-SNE is computationally expensive
## by far the most popular technique for visualization and most successful in practice
from sklearn.manifold import TSNE
n_components = 2
learning_rate = 300 ## The learning rate should be adjusted for different datasets
perplexity = 30 ## The perplexity is related to the number of nearest neighbors that is used in other manifold learning algorithms. Larger datasets usually require a larger perplexity.
## perplexity is useually between 5 and 50
n_iter = 5000 ## The number of iterations should be adjusted depending on the dataset.
early_exaggeration = 12 ## The early exaggeration factor is used to increase the space between clusters and allows t-SNE to find more distinct clusters.
random_state = 42

tsne = TSNE(n_components=n_components, learning_rate=learning_rate, perplexity=perplexity, n_iter=n_iter, early_exaggeration=early_exaggeration, random_state=random_state)

X_train_tsne = tsne.fit_transform(X_train_pca.loc[0:1000 , :9])
X_train_tsne = pd.DataFrame(X_train_tsne , index=train_index[0:1001])
scatterPlot(X_train_tsne, y_train, 't-SNE')


################################# Independent Component Analysis (ICA) ########################################
## ICA is a linear dimensionality reduction technique that is used to separate independent sources from a mixture of signals.
## ICA is used in signal processing and feature extraction. It is also used to remove noise from images.
## ICA is computationally expensive
##  ICA is used to separate independent sources from a mixture of signals.
from sklearn.decomposition import FastICA
n_components = 30 ## The number of components should be less than the number of features.
algorithm = 'parallel'
whiten = 'unit-variance'
max_iter = 200
random_state = 42

fastICA = FastICA(n_components=n_components, algorithm=algorithm, whiten=whiten, max_iter=max_iter, random_state=random_state)
fastICA.fit_transform(X_train)  ## to speed up, fit only a subset of the data then transform the whole data
X_train_fastICA = fastICA.transform(X_train)
X_train_fastICA = pd.DataFrame(X_train_fastICA , index=train_index)

scatterPlot(X_train_fastICA, y_train, 'Fast ICA')