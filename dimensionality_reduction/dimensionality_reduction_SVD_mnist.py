# SVD is more efficient since it foregoes the calculation of the covariance matrix. 
# However, at least for mnnist dataset, PCA creates a better separation of the classes.
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
from sklearn.random_projection import GaussianRandomProjection

n_components = 'auto'
eps = 0.5
random_state = 42

GRP = GaussianRandomProjection(n_components=n_components, eps=eps, random_state=random_state)

X_train_GRP = GRP.fit_transform(X_train)
X_train_GRP = pd.DataFrame(X_train_GRP)

scatterPlot(X_train_GRP, y_train, 'GRP')