import umap

class myUMAP:
	"""
		class method to reduce the data to 2D or 3D using Umap.
	"""
	def __init__(self) -> None:
		pass

	def umap(self, target_data:any, dims=2):
		"""
			Performs Umap on the target data.

			Args:
				target_data: The data to be reduced.
				dims: The number of dimensions to reduce the data to.
			Returns:
				The reduced data.
		"""
		if dims==2:
			# Using Umap for dimension reduction(2D)
			umap_model = umap.UMAP(n_components=2)
			reduction_data = umap_model.fit_transform(target_data)
			# for plotting. split into x and y
			X_data = reduction_data[:,0]
			Y_data = reduction_data[:,1]

			return (X_data, Y_data)
		elif dims==3:
			# Using PCA for dimension reduction(3D)
			umap_model = umap.UMAP(n_components=2)
			reduction_data = umap_model.fit_transform(target_data)
			# for plotting. split into x and y
			X_data = reduction_data[:,0]
			Y_data = reduction_data[:,1]
			Z_data = reduction_data[:,2]

			return (X_data, Y_data, Z_data)
		else:
			raise ValueError("Dimension must be 2 or 3")