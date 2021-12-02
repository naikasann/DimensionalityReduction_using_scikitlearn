import sklearn
import sklearnex
from sklearn.decomposition import PCA
from sklearnex import patch_sklearn

class myPCA:
	"""
		class method to reduce the data to 2D or 3D using PCA.
	"""
	def __init__(self) -> None:
		patch_sklearn()

	def pca(self, target_data:any, dims=2):
		"""
			Performs PCA on the target data.

			Args:
				target_data: The data to be reduced.
				dims: The number of dimensions to reduce the data to.
			Returns:
				The reduced data.
		"""
		if dims == 2:
			# Using pca for dimension reduction(2D)
			tsne = PCA(n_components=2, random_state=1234)
			reduction_data = tsne.fit_transform(target_data)
			# for plotting. split into x and y
			X_data = reduction_data[:,0]
			Y_data = reduction_data[:,1]

			return (X_data, Y_data)
		elif dims == 3:
			# Using pca for dimension reduction(3D)
			tsne = PCA(n_components=3, random_state=1234)
			reduction_data = tsne.fit_transform(target_data)
			# for plotting. split into x and y and z
			X_data = reduction_data[:,0]
			Y_data = reduction_data[:,1]
			Z_data = reduction_data[:,2]

			return (X_data, Y_data, Z_data)
		else:
			raise ValueError("Dimension must be 2 or 3")

