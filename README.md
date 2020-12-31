# ColorQuantization
K-means for Color Quantization


K-means clustering algorithm:

Unsupervised machine learning algorithm where we start clustering data based on k
clusters as we don’t have true label to validate that so we have to know the dataset thoroughly to
start with optimal k. It’s hard clustering algorithm where one data point belongs to only one cluster
by enhancing K-means with probabilistic approach where we can predict the probability of the
data point belonging to the specific cluster there by achieving the soft clustering.


Color Quantization:

Process of reducing the number of distinct colors used in the image and the
new image should be visually similar to the original image. Its used in image compression. Here
we use k-means algorithm to perform color quantization of the given image.
We have to find the k colors to represent the image and value of k varies from 3,5,10,20. Use those
k-colors to create image which will be compressed version of the image.
For given image we have – 347633 unique colors representing the image(RGB encoding 3
dimensional). We have to reduce number of colors to represent image with just k colors and to id
entify these k colors we will be employing k means algorithm.

