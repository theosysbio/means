import numpy as np
def _label_axes(ax, x_label, y_label, fontsize=20, rotate_x_ticks=True):
    ax.set_xlabel(x_label, fontsize=fontsize)
    ax.set_ylabel(y_label, fontsize=fontsize)
    if rotate_x_ticks:
        from matplotlib.artist import setp
        setp(ax.get_xticklabels(), rotation=90)

def plot_contour(x, y, z, x_label, y_label, ax=None, *args, **kwargs):

    from matplotlib import pyplot as plt
    from matplotlib.mlab import griddata

    if ax is None:
        ax = plt.gca()

    x = np.array(x, dtype=np.float)
    y = np.array(y, dtype=np.float)
    z = np.array(z, dtype=np.float)

    # Remove duplicates from x, y, z
    seen_values = {}
    for xi, yi, zi in zip(x, y, z):
        try:
            seen_values[xi, yi].append(zi)
        except KeyError:
            seen_values[xi, yi] = [zi]

    new_x, new_y, new_z = [], [], []
    for (xi, yi), zi_list in seen_values.iteritems():
        new_x.append(xi)
        new_y.append(yi)

        # Use mean of distances. TODO: is this the right thing to do?
        new_z.append(np.mean(zi_list))
    new_x, new_y, new_z = np.array(new_x, dtype=np.float), np.array(new_y, dtype=np.float), \
                          np.array(new_z, dtype=np.float)

    # Replace x, y, z with new_x, new_y, new_z
    x, y, z = new_x, new_y, new_z

    xi = np.linspace(np.min(x), np.max(x), 100)
    yi = np.linspace(np.min(y), np.max(y), 100)

    # Interpolate points to a grid
    try:
        zi = griddata(x, y, z, xi, yi)
    except Exception as e:
        raise Exception("Got {0!r} while interpolating the landscape."
                        "this may be due to `matplotlib.delaunay` package not being able to"
                        "handle the interpolation. Try installing natgrid package via pip: "
                        "`pip install git+git://github.com/matplotlib/natgrid.git`".format(e))

    # Plot contours
    ax.contourf(xi, yi, zi, *args, **kwargs)
    cs = ax.contour(xi, yi, zi, colors='k')

    # Some labels for contour lines
    ax.clabel(cs, inline=True)

    _label_axes(ax, '${0}$'.format(x_label), '${0}$'.format(y_label), fontsize=20, rotate_x_ticks=True)

def plot_2d_trajectory(x, y,
                       x_label='', y_label='',
                       legend=False,
                       ax=None,
                       start_and_end_locations_only=False,
                       start_marker='bo',
                       end_marker='rx',
                       *args, **kwargs):

    if ax is None:
        from matplotlib import pyplot as plt
        ax = plt.gca()

    if not start_and_end_locations_only:
        ax.plot(x, y, *args, **kwargs)

    x = np.array(x)
    y = np.array(y)

    max_x = np.max(x)
    min_x = np.min(x)
    padding_x = (max_x - min_x) * 0.1 / 2.0

    max_y = np.max(y)
    min_y = np.min(y)
    padding_y = (max_y - min_y) * 0.1 / 2.0


    ax.set_xlim(min(x)-padding_x, max(x)+padding_x)
    ax.set_ylim(min(y)-padding_y, max(y)+padding_y)

    ax.plot(x[0], y[0], start_marker, label='Start')
    ax.plot(x[-1], y[-1], end_marker, label='End')

    _label_axes(ax, '${0}$'.format(x_label), '${0}$'.format(y_label), fontsize=20, rotate_x_ticks=True)

    if legend:
        ax.legend()

