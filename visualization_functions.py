import matplotlib.pyplot as plt

def single_plot(image, colors="Greys_r", plot_title="Resulting Image", xlabel="x", ylabel="y", output_dir=None, save_plot=False):
    """
    This function generates a single plot and can optionally save the result to a file.
    """
    # Display plot
    plt.imshow(image, cmap=colors)
    plt.title(plot_title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()

    # Optional - save plot to file
    if save_plot == True:
        plt.savefig(output_dir)


def multi_axis_plot(image):
    pass