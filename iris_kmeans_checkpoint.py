from sklearn import datasets
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""The sepal and petal lengths and widths are in an array called iris.data. The species classifications for each of the 150 samples is in another array called iris.target."""

iris = datasets.load_iris()
print (iris.data)
print (iris.target)

iris

"""Let’s convert our arrays to a pandas DataFrame for ease of use. I am setting the column names explicitly."""

x = pd.DataFrame(iris.data, columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
y = pd.DataFrame(iris.target, columns=['Target'])

y

"""visualise the data in a scatter plot to see if there is any pattern visible."""

plt.figure(figsize=(12,3))

colors = np.array(['red', 'green', 'blue'])

plt.subplot(1, 2, 1)
plt.scatter(x['Sepal Length'], x['Sepal Width'], c=colors[y['Target']], s=40)
plt.title('Sepal Length vs Sepal Width')

plt.subplot(1,2,2)
plt.scatter(x['Petal Length'], x['Petal Width'], c= colors[y.Target], s=40)
plt.title('Petal Length vs Petal Width')

"""We can clearly see the grouping in the plots with the red dots, which correspond to species Setosa. The green and blue dots are not so clearly separable.

Now let’s use the KMeans algorithm to see if it can create the clusters automatically.
"""

model = KMeans(n_clusters=3)
model.fit(x)  #function runs the algo on the data and creates the clusters. Each sample in the dataset is then assigned a cluster id (0, 1, 2, etc).

print (model.labels_) #model.labels_ holds the array of the cluster ids, so let’s take a look at it.

"""An important note: iris.target is an array of integers used to represent the Iris species. 0=Setosa, 1=Versicolor, 2=Virginica. And the KMeans model object also assigns integer ids for the three clusters (n_clusters =3 above), namely 0, 1, 2. Its important to note that the KMeans model has no knowledge of the iris.target data, and the clusters being given ids 0,1,2 is just a coincidence."""

#Start with a plot figure of size 12 units wide & 3 units tall
plt.figure(figsize=(12,3))

# Create an array of three colours, one for each species.
colors = np.array(['red', 'green', 'blue'])

# The fudge to reorder the cluster ids.
predictedY = np.choose(model.labels_, [1, 0, 2]).astype(np.int64)

# Plot the classifications that we saw earlier between Petal Length and Petal Width
plt.subplot(1, 2, 1)
plt.scatter(x['Petal Length'], x['Petal Width'], c=colors[y['Target']], s=40)
plt.title('Before classification')

# Plot the classifications according to the model
plt.subplot(1, 2, 2)
plt.scatter(x['Petal Length'], x['Petal Width'], c=colors[predictedY], s=40)
plt.title("Model's classification")
