import yaml
import argparse
from sklearn.decomposition import PCA
from matplotlib import animation
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description="")
parser.add_argument('-dim', '--arg1')
parser.add_argument('-config', default='config.yaml')

# parser loaded.
args = parser.parse_args()
# Load the configuration file.
with open(args.config, "r") as yaml_file:
    config = yaml.safe_load(yaml_file)

# Defined for animation of 3D graphs.
# I'm definitely doing a bad job of writing this. ;(
# FIXME : global variable => local variable.
if args.arg1 == "3D":
    gif = plt.figure()
    gif_ax = gif.add_subplot(111, projection='3d')

def dataload(datapath, attribute_number):
    # read csv data(terget data)
    load_df = pd.read_csv(datapath)
    # Extracting data
    X_data = load_df[load_df.columns[load_df.columns != load_df.columns[attribute_number]]].values
    Y_data = load_df[load_df.columns[attribute_number]]

    print("data data : {}".format(X_data.shape[0]))

    return X_data, Y_data

def PCA_2D(X_data, Y_data, output_directory):
    # Using PCA for dimension reduction(2D)
    pca = PCA(n_components=2)
    X_embedded = pca.fit_transform(X_data)

    # Create a two-dimensional scatter plot.
    fig = plt.figure()
    # plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=Y_data, cmap=cm.tab10)

    plot_xlist = [[] for _ in range(4)]
    plot_ylist = [[] for _ in range(4)]
    for cnt, y in enumerate(Y_data):
        plot_xlist[y].append(X_embedded[cnt][0])
        plot_ylist[y].append(X_embedded[cnt][1])
    for cnt, (x, y) in enumerate(zip(plot_xlist, plot_ylist)):
        plt.scatter(x, y, cmap=cm.tab10)

    plt.title("PCA result")
    plt.colorbar()
    plt.legend(loc='best', fontsize = 10, labels=["oldwalk", "slowwalk", "run", "normalwalk"])
    # save figure.
    fig.savefig(output_directory + "PCA_result_2D.png")

# for gif image.(gif animation setting.)
def animation_for_3D(i):
    print("\r Gif images are created... {}/{} | {} %".format(i , config["gif_frame"], i/config["gif_frame"]),end="")
    gif_ax.view_init(elev=10., azim=i)
    return gif,

def PCA_3D(X_data, Y_data, output_directory):
    # Using PCA for dimension reduction(3D)
    pca = PCA(n_components=3)
    X_embedded = pca.fit_transform(X_data)

    # Create a three-dimensional scatter plot.
    mappable = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2], c=Y_data, cmap=cm.tab10)
    plt.title("PCA result")
    plt.legend(fontsize = 10)
    gif.colorbar(mappable, ax=gif_ax)

    # Create a Gif image. (This will take some time.)
    anim = animation.FuncAnimation(gif, animation_for_3D,
                                    frames=config["gif_frame"],
                                    interval=config["gif_interval"],
                                    blit=True)
    # Gif image save.
    anim.save(output_directory + "PCA_result_3D.gif", writer="imagemagick")

def main():
    print("=== PCA ===")
    X_data, Y_data = dataload(config["datapath"], config["attribute_number"])

    # Execute according to the target dimension (command line argument)
    if args.arg1 == "3D":
        PCA_3D(X_data, Y_data, config["output_directory"])
    elif args.arg1 == "2D":
        PCA_2D(X_data, Y_data, config["output_directory"])
    else:
        # Default is to do it in 2D!
        PCA_2D(X_data, Y_data, config["output_directory"])

    # Whether to display or not
    if config["isshow"]:
        plt.show()

    print("program end.")
    print("\n===========")

if __name__ == "__main__":
    main()