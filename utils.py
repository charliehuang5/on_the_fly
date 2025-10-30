import plotly.graph_objects as go

def interactive_plot(x, y, z, color_list=None, title='', marker_size=2, cmap='Spectral_r', vmin=None, vmax=None):
    """
    Creates an interactive 3D scatter plot using Plotly.

    Parameters:
        x (array-like): The x-coordinates of the points.
        y (array-like): The y-coordinates of the points.
        z (array-like): The z-coordinates of the points.
        color_list (array-like, optional): List or array of colors for each point. If None, colors are mapped to `z` values.
        title (str, optional): Title of the plot. Defaults to ''.
        marker_size (int or float, optional): Size of the markers. Defaults to 2.
        cmap (str, optional): Colormap to use for coloring the points. Defaults to 'Spectral_r'.
        vmin (float, optional): Minimum value for color scale.
        vmax (float, optional): Maximum value for color scale.

    Returns:
        plotly.graph_objs._figure.Figure: The interactive 3D scatter plot figure.
    """
    # Create interactive plot
    use_color_list = color_list is not None
    cmin = vmin if vmin is not None else (min(color_list) if use_color_list else min(z))
    cmax = vmax if vmax is not None else (max(color_list) if use_color_list else max(z))

    marker_settings = dict(
        size=marker_size,
        color=color_list if use_color_list else z,
        opacity=0.5,
        colorscale=cmap,
        colorbar=dict(
            title="Color",
            thickness=20,
            len=0.75,            
            x=1.05  # Push it to the right a bit
        ),
        cmin=cmin,
        cmax=cmax
    )

    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=marker_settings
    )])

    fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig

def add_to_plot(fig, x, y, z, color_list=None, opacity=0.1, cmap='Spectral_r', marker_size=2):
    # Add another scatter3d trace to the existing figure
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z, mode='markers',
        marker=dict(size=marker_size, color=color_list if color_list is not None else z,
                    opacity=opacity, colorscale=cmap)
    ))
    return fig  # Return the updated figure

def get_color_list():
    colors = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#7f7f7f',  # gray
    '#bcbd22',  # yellow-green
    '#17becf',  # cyan
    '#393b79',  # dark blue
    '#637939',  # olive green
    '#8c6d31',  # dark orange
    '#843c39',  # dark red
    '#7b4173',  # dark purple
    '#3182bd',  # steel blue
    '#e6550d',  # rust orange
    '#31a354',  # forest green
    '#756bb1',  # medium purple
    '#636363'   # dark gray
    ]
    return colors