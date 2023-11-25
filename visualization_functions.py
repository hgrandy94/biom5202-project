import matplotlib.pyplot as plt

def single_plot(image, colors="Greys_r", plot_title="Resulting Image", xlabel="x", ylabel="y", output_dir=None, save_plot=False):
    """
    This function generates a single plot and can optionally save the result to a file.
    """
    # Initialize plot
    fig = plt.plot(figsize=(12,6))
    
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
    
    return fig

# Function created with assistance from ChatGPT
def double_plot(img1, img2, color1='blue', color2='orange', title='', xlabel='', ylabel='', output_dir=None, save_plot=False):
    """
    Create a custom plot with two subplots.

    Parameters:
    - data1: Data for the first subplot.
    - data2: Data for the second subplot.
    - color1: Color for the first subplot (default: 'blue').
    - color2: Color for the second subplot (default: 'orange').
    - title: Title of the plot (default: '').
    - xlabel: Label for the x-axis (default: '').
    - ylabel: Label for the y-axis (default: '').

    Returns:
    - fig: The created Matplotlib figure.
    """

    # Create a figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot data in the first subplot
    ax1.plot(img1, color=color1)
    ax1.set_title('Plot 1')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)

    # Plot data in the second subplot
    ax2.plot(img2, color=color2)
    ax2.set_title('Plot 2')
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)

    # Set the overall title for the entire plot
    fig.suptitle(title)

    # Adjust layout for better spacing
    plt.tight_layout()

    # Optional - save plot to file
    if save_plot == True:
        plt.savefig(output_dir)

    # Return the created figure
    return fig
