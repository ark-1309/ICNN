import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def plot_images(batch, n_images=10):
    fig, axes = plt.subplots(1, n_images, figsize=(n_images, 1), dpi=100)
    for i in range(n_images):
        axes[i].imshow(batch[i, 0], cmap='gray')
        axes[i].set_xticks([]); axes[i].set_yticks([])
    fig.tight_layout(pad=0.1)
    return fig

def fig2data (fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ()
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring (fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll (buf, 3, axis = 2)
    return buf

def fig2img (fig):
    buf = fig2data (fig)
    w, h, d = buf.shape
    return Image.frombytes("RGBA", (w ,h), buf.tostring())

