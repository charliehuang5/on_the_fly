import numpy as np
from matplotlib import pyplot as plt


def downsample(input_array, downscale_factor: tuple = (0.25, 0.25, 0.25), order=3):
    """
    Downsample a 3D NumPy array
    """
    from scipy.ndimage import zoom
    new_array = zoom(input_array, downscale_factor, order=order)
    return new_array


def create_movie_from_slices(
    array, output_file="output_movie.mp4", fps=15, cmap="Oranges"
):
    """
    Create a movie from an array of shape (xRes, yRes, zSlice) by plotting each zSlice.

    Parameters:
    - array: 3D numpy array with shape (xRes, yRes, zSlice)
    - output_file: Path to save the output movie file, e.g., "output_movie.mp4"
    - fps: Frames per second for the output video
    - cmap: Colormap for the plot, e.g., "viridis", "gray", etc.
    """
    import matplotlib.animation as animation
    from matplotlib.colors import ListedColormap

    array = array.copy()
    array = np.moveaxis(array, 1, 0)

    # Determine the resolution of each frame
    xRes, yRes, zSlices = array.shape

    # Create a modified "Oranges" colormap with black as the lowest color
    cmap = plt.cm.get_cmap(cmap)
    cmap.set_bad(color="black")  # Set NaNs or invalid values to appear colored

    # Set up the figure for plotting
    fig, ax = plt.subplots()
    ax.axis("off")
    # cax = ax.imshow(array[:, :, 0], cmap=cmap, vmin=0, vmax=np.max(array))
    cax = ax.imshow(array[:, :, 0], cmap=cmap, vmin=np.nanmin(array), vmax=np.nanmax(array))

    def update(z):
        """Update function for each frame"""
        cax.set_array(array[:, :, z])
        return (cax,)

    # Create the animation object
    ani = animation.FuncAnimation(fig, update, frames=zSlices, blit=True, repeat=False)

    # Save as an MP4 movie
    ani.save(output_file, fps=fps, extra_args=["-vcodec", "libx264"])
    plt.close(fig)
    print(f"Movie saved to {output_file}")


def makeResponderMovie(roiIdxs, supervoxels, fda, output_file, debug=False):
    fdaWorking = fda.copy()
    fdaWorking = 255 * (fdaWorking - np.min(fdaWorking)) / np.max(fdaWorking)
    supervoxelsWorking = np.copy(supervoxels)
    supervoxelMask = np.isin(supervoxelsWorking, roiIdxs)
    if debug:
        import pdb

        pdb.set_trace()
    fdaWorking[supervoxelMask] = np.nan
    create_movie_from_slices(array=fdaWorking, output_file=output_file)
