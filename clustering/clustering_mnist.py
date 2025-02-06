import pandas as pd
import gzip
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as pp
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import average_precision_score, precision_recall_curve




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

file_path = '..//../datasets_unsupervised_aapatel/mnist_data/mnist.pkl.gz'
f = gzip.open(file_path , 'rb')
train_set, validation_set, test_set = pickle.load(f, encoding='latin1')
f.close()

color = sns.color_palette()

x_train, y_train = train_set[0], train_set[1]
x_validation, y_validation = validation_set[0], validation_set[1]
x_test, y_test = test_set[0], test_set[1]

train_index = range(0, len(x_train))
validation_index = range(len(x_train), len(x_train) + len(x_validation))
test_index = range(len(x_train) + len(x_validation), len(x_train) + len(x_validation) + len(x_test))

x_train = pd.DataFrame(data=x_train, index=train_index)
y_train = pd.Series(data=y_train, index=train_index)

x_validation = pd.DataFrame(data=x_validation, index=validation_index)
y_validation = pd.Series(data=y_validation, index=validation_index)

x_test = pd.DataFrame(data=x_test, index=test_index)
y_test = pd.DataFrame(data=y_test, index=test_index)


from sklearn.decomposition import PCA

n_components = 784
whiten = False
random_state = 2018

pca = PCA(n_components=n_components, whiten=whiten, random_state=random_state)
X_train_PCA = pca.fit_transform(x_train)
X_train_PCA = pd.DataFrame(data=X_train_PCA, index=x_train.index)


######################### k-means clustering ############################
from sklearn.cluster import KMeans
n_clusters = 10
n_init = 10
max_iter = 300
tol = 0.0001
random_state = 2018
n_jobs = 2

kMeans_inertia = pd.DataFrame(data=[], index=range(2, 21), columns=['inertia'])

for n_clusters in range(2, 21):
    kMeans = KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter, tol=tol, random_state=random_state)
    kMeans.fit(X_train_PCA)
    kMeans_inertia.loc[n_clusters] = kMeans.inertia_
    kMeans_inertia.plot()