import argparse
from re import A
from mypackage.dataloader.loader import loader
from mypackage.dimensionality_reduction.pca import myPCA
from mypackage.dimensionality_reduction.tsne import myTSNE
from mypackage.dimensionality_reduction.umap import myUMAP
from mypackage.Ploter.plot import myPloter

def argparser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--input', '-i', type=str, default='input.csv')
	parser.add_argument('--output', '-o', type=str, default='output.png')
	parser.add_argument("--method", "-m", type=str, default="tsne")
	parser.add_argument("--dim", "-d", type=int, default=2)
	parser.add_argument('--input_num', '-in', type=int, default=10)
	parser.add_argument('--category', '-c', type=str, default='config/category.yaml')
	return parser.parse_args()

def main():
	# load args
	args = argparser()

	# load dataset.
	X_data, Y_data = loader.csv_dataload(args.input, args.input_num)
	print("X_data:", len(X_data))
	# load category.
	category = loader.category_dataload(args.category)
	print("category:", category)

	# dimensionality reduction.
	if args.method == "pca":
		mypca = myPCA()
		reduction_data = mypca.pca(X_data, args.dim)
	elif args.method == "tsne":
		mytsne = myTSNE()
		reduction_data = mytsne.tsne(X_data, args.dim)
	elif args.method == "umap":
		umap = myUMAP()
		reduction_data = umap.umap(X_data, args.dim)
	else:
		raise ValueError("method is not supported.")

	# plot.
	myPloter.plot(reduction_data, Y_data, category, args.output)

if __name__ == '__main__':
	main()