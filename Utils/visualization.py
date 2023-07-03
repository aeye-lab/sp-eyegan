import numpy as np
from tqdm import tqdm
import seaborn as sns
import pandas as pd
from matplotlib import image
from matplotlib import pyplot

def vel_to_dva(vel_data, x_start = 0,
             y_start = 0):
    x_vel = vel_data[:,0]
    y_vel = vel_data[:,1]
    x_px  = []
    y_px  = []
    cur_x_pos = x_start
    cur_y_pos = y_start
    for i in range(len(x_vel)):
        x_px.append(cur_x_pos + x_vel[i])
        y_px.append(cur_y_pos + y_vel[i])
        cur_x_pos = x_px[-1]
        cur_y_pos = y_px[-1]
    return np.concatenate([np.expand_dims(np.array(x_px),axis=1),
                           np.expand_dims(np.array(y_px),axis=1)],axis=1)



def draw_display(dispsize, imagefile=None):
    # construct screen (black background)
    # dots per inch
    img = image.imread(imagefile)
    dpi = 100.0
    # determine the figure size in inches
    figsize = (dispsize[0]/dpi, dispsize[1]/dpi)
    # create a figure
    fig = pyplot.figure(figsize=figsize, dpi=dpi, frameon=False)
    ax = pyplot.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)
    # plot display
    ax.axis([0, dispsize[0], 0, dispsize[1]])
    ax.imshow(img)

    return fig, ax




def pix2deg(pix, screenPX,screenCM,distanceCM, adjust_origin=True):
    # Converts pixel screen coordinate to degrees of visual angle
    # screenPX is the number of pixels that the monitor has in the horizontal
    # axis (for x coord) or vertical axis (for y coord)
    # screenCM is the width of the monitor in centimeters
    # distanceCM is the distance of the monitor to the retina 
    # pix: screen coordinate in pixels
    # adjust origin: if origin (0,0) of screen coordinates is in the corner of the screen rather than in the center, set to True to center coordinates
    pix=np.array(pix)
    # center screen coordinates such that 0 is center of the screen:
    if adjust_origin: 
        pix = pix-(screenPX)/2 # pixel coordinates start with (0,0) 
    # eye-to-screen-distance in pixels of screen
    distancePX = distanceCM*(screenPX/screenCM)
    return np.arctan2(pix,distancePX) * 180/np.pi #  *180/pi wandelt bogenmass in grad


def deg2pix(deg, screenPX, screenCM, distanceCM, adjust_origin = True, offsetCM = 0):
    # Converts degrees of visual angle to pixel screen coordinates
    # screenPX is the number of pixels that the monitor has in the horizontal
    # screenCM is the width of the monitor in centimeters
    # distanceCM is the distance of the monitor to the retina 
    phi = np.arctan2(1,distanceCM)*180/np.pi
    pix = deg/(phi/(screenPX/(screenCM)))
    if adjust_origin:
        pix += (screenPX/2)
    if offsetCM != 0:
        offsetPX = offsetCM*(screenPX/screenCM)
        pix += offsetPX
    return pix


def plot_scanpath_on_image(image_path,
                           x_locations,
                           y_locations,
                           expt_txt,
                           ):
    fig, ax = draw_display(dispsize=(expt_txt['px_x'], expt_txt['px_y']), imagefile=image_path)    

    ax.plot(np.array(x_locations)/expt_txt['max_dva_x'] * expt_txt['px_x'],
            np.array(y_locations)/expt_txt['max_dva_y'] * expt_txt['px_y'],'r-')

    ax.invert_yaxis()
    pyplot.show()