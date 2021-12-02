import matplotlib.pyplot as plt
import matplotlib.cm as cm

class myPloter:
	def plot(plot_data:any, y_data:any, label, title):
		if len(plot_data) == 2:
			fig = plt.figure()
			plt.scatter(plot_data[0], plot_data[1], c=y_data, cmap=cm.tab10)

			plt.title(title)
			plt.colorbar()
			plt.legend(loc='best', fontsize = 10, labels=label)
			# save figure.
			fig.savefig("Umap_result_2D.png")
		elif len(plot_data) == 3:
			fig = plt.figure()
			ax = fig.add_subplot(111, projection='3d')
			ax.scatter(plot_data[0], plot_data[1], plot_data[2], c=y_data, cmap=cm.tab10)
			ax.legend(loc='best', fontsize = 10, labels=label)
			# save figure.
			fig.savefig("Umap_result_3D.png")
			plt.show()
		else:
			raise Exception("The length of plot_data is not 2 or 3.")