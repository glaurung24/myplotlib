# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 17:14:29 2017

@author: andi
"""

from __future__ import division
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import Rectangle
#from matplotlib import rc

#plt.rc('font',**{'family':'serif','serif':['Palatino']})

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


labelsize=20
titlesize=24
ticklabelsize=18
legendfontsize=18
fontsize=18
linewidth=2.0
tickwidth=1.0
majorticksize=8.0
minorticksize=4.0
markersize = 7.5

plotheight = 10*0.6
plotwidth=16.1 * 0.6

fig = []
ax = []
aspect=1.33
eps=1E-30

x_label=r'$x$'
y_label=r'$y$'
z_label=r'$z$'

almost_black = '#262626'
red = 'crimson'
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

colorcounter=0
useHigherColorContrast=False

for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)

colorscheme=tableau20

lines=[]
legend_texts=[]
plotstyle = 'default'

cbar = []
heatmap = []
cmap=mpl.cm.viridis
c_max = 0
c_min = 0

size_set=False
legendartist=False

def frexp10(x):
    exp = int(np.floor(np.log10(x)))
    return x / 10**exp, exp

def getTickSpacing(c_min, c_max):

    mantissa, exponent = frexp10(c_max)
    int_max = np.ceil(c_max*10.0**-exponent)
    int_min = np.floor(c_min*10.0**-exponent)


    nr_ticks=[0,6,5,7,5,6,7,8,5,7,6]

    distance=int(int_max-int_min)
    min_nr = float(int_min)*10.0**exponent
    max_nr = float(int_max)*10.0**exponent
    tickpos=np.linspace(min_nr, max_nr, nr_ticks[distance])
    return tickpos


def setAspectRatio(ratio):
    if ratio == 'square':
        ax.set_aspect('equal')
    else:
        ax.set_aspect(ratio)


def xlabel(label):
    global x_label
    x_label = label
    ax.set_xlabel(label)

def ylabel(label):
    global y_label
    y_label = label
    ax.set_ylabel(label)

def zlabel(label):
    global z_label
    z_label = label
    ax.set_zlabel(label)



def setFigureSize(size):
    global size_set
    if size == 'small' and not size_set:
        scale=1.5
        global labelsize
        global titlesize
        global ticklabelsize
        global legendfontsize
        global fontsize
        labelsize= int(labelsize * scale)
        titlesize=int(titlesize * scale)
        ticklabelsize=int(ticklabelsize * scale)
        legendfontsize=int(legendfontsize * scale)
        fontsize=int(fontsize * scale)
        size_set=True
    if size == 'medium' and not size_set:
        scale=1.3
        global labelsize
        global titlesize
        global ticklabelsize
        global legendfontsize
        global fontsize
        labelsize= int(labelsize * scale)
        titlesize=int(titlesize * scale)
        ticklabelsize=int(ticklabelsize * scale)
        legendfontsize=int(legendfontsize * scale)
        fontsize=int(fontsize * scale)
        size_set=True


def colorbar(*args, **kwarg):
    global cbar

    tickpos=getTickSpacing(c_min, c_max)

    if not kwarg:
        kwarg = dict(kwarg, ticks=tickpos)
    if 'ticks' not in kwarg:
        kwarg = dict(kwarg, ticks=tickpos)
    cbar = plt.colorbar(heatmap, *args, **kwarg)

    cbar.formatter.set_scientific(True)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.ax.yaxis.set_offset_position('right')
    cbar.ax.tick_params(labelsize=ticklabelsize, color=almost_black)

    cbar.update_ticks()

def colorbarLabel(label):
    cbar.set_label(label, color=almost_black, fontsize=labelsize) #, rotation=270)



def initPlot(plstyle='default'):
    global plotstyle
    plotstyle=plstyle
    #Mainly taken from http://www.randalolson.com/2014/06/28/how-to-make-beautiful-data-visualizations-in-python-with-matplotlib/


    plt.rc('text', usetex=True)

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Palatino'
    plt.rcParams['font.monospace'] = 'Ubuntu Mono'
    plt.rcParams['font.size'] =  fontsize
    plt.rcParams['axes.labelsize'] = labelsize
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = titlesize
    plt.rcParams['xtick.labelsize'] = ticklabelsize
    plt.rcParams['ytick.labelsize'] = ticklabelsize
    plt.rcParams['legend.fontsize'] = legendfontsize
    plt.rcParams['lines.linewidth'] = linewidth
    plt.rcParams['lines.markersize'] = markersize


    mpl.rcParams['xtick.major.size'] = majorticksize
    mpl.rcParams['xtick.major.width'] = tickwidth
    mpl.rcParams['xtick.minor.size'] = minorticksize
    mpl.rcParams['xtick.minor.width'] = tickwidth

    mpl.rcParams['ytick.major.size'] = majorticksize
    mpl.rcParams['ytick.major.width'] = tickwidth
    mpl.rcParams['ytick.minor.size'] = minorticksize
    mpl.rcParams['ytick.minor.width'] = tickwidth

#    mpl.rcParams['image.cmap'] = 'viridis'

    global ax
    global fig
    global colorcounter
    global cmap

    plt.close()
    colorcounter=0

    if plotstyle is 'default':
        fig = plt.figure(figsize=(plotwidth, plotheight))
        ax=fig.add_subplot(111)

        axisgrey=(0.7,0.7,0.7)
        #        ax.spines["top"].set_visible(False)
        ax.spines["top"].set_color(axisgrey)
#        ax.spines["bottom"].set_visible(False)
#        ax.spines["right"].set_visible(False)
        ax.spines["right"].set_color(axisgrey)
#        ax.spines["left"].set_visible(False)

        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

        plt.yticks(fontsize=ticklabelsize)
        plt.xticks(fontsize=ticklabelsize)

        plt.xlabel(x_label)
        plt.ylabel(y_label)

        #make the black less black and easier on the eyes


        ax.xaxis.label.set_color(almost_black)
        ax.yaxis.label.set_color(almost_black)
        ax.spines['left'].set_color(almost_black)
        ax.spines['bottom'].set_color(almost_black)
        ax.title.set_color(almost_black)


        cmap=mpl.cm.bone_r
        return


    if plotstyle is 'bmh':
        plt.style.use('bmh')
        return

    if plotstyle is '3D_plain':
        fig = plt.figure(figsize=(10, 10))
        ax=fig.add_subplot(111, projection='3d')

        ax.set_axis_off()
        ax.patch.set_alpha(0.0)
        return
    if plotstyle is '3D':
        fig = plt.figure(figsize=(10, 10))
        ax=fig.add_subplot(111, projection='3d')
#        ax.patch.set_alpha(0.0)
        return
    if plotstyle is 'descriptive':
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for side in ['bottom','right','top','left']:
            ax.spines[side].set_visible(False)
        for side in ['bottom', 'left']:
            ax.spines[side].set_position('zero')

#        ax.xaxis.set_ticks_position('none') # tick markers
#        ax.yaxis.set_ticks_position('none')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.tick_params(axis='x', which='bottom', width=1.5, length=10.0, direction='both', pad=4.5, labelsize=labelsize)
        ax.tick_params(axis='y', which='left', width=1.5, length=10.0, direction='both', pad=0, labelsize=labelsize)



def plot_surface(*args, **kwargs):
    global colorcounter
    if not kwargs:
        kwargs= {'color':getColor()}
    if 'color' in kwargs:
        ax.plot_surface(*args, **kwargs)
    else:
        if 'cmap' in kwargs:
            ax.plot_surface(*args, **kwargs)
        else:
            ax.plot_surface(*args, **dict(kwargs, color=getColor()))


def contour(*args, **kwargs):
    global colorcounter
    if not kwargs:
        kwargs= {'color':getColor()}
    if 'cmap' not in kwargs:
        kwargs = dict(kwargs, cmap=cmap )
    if 'color' in kwargs:
        ax.contour(*args, **kwargs)
    else:
        ax.contour(*args, **dict(kwargs, color=getColor()))

def plot_trisurf(*args, **kwargs):
    global colorcounter
    if not kwargs:
        kwargs= {'color':getColor()}
    if 'color' in kwargs:
        ax.plot_trisurf(*args, **kwargs)
    else:
        if 'cmap' in kwargs:
            ax.plot_trisurf(*args, **kwargs)
        else:
            ax.plot_trisurf(*args, **dict(kwargs, color=getColor()))

def pcolormesh(*args, **kwargs):
    global heatmap
    global ax
    global c_max
    global c_min
    if not kwargs:
        kwargs = dict(cmap=cmap)
#    if 'shading' not in kwargs:
#        kwargs = dict(kwargs, shading='gouraud')
    if 'cmap' not in kwargs:
        kwargs = dict(kwargs, cmap=cmap )
    if kwargs.pop('opt_range', False):
        c_max = np.max(args[2])
        c_min = np.min(args[2])
        ticks=getTickSpacing(c_min, c_max)
        vmin=ticks[0]
        vmax=ticks[-1]
        kwargs = dict(kwargs, vmin=vmin )
        kwargs = dict(kwargs, vmax=vmax )
        heatmap =  ax.pcolormesh(*args, **kwargs)
    else:
        heatmap = ax.pcolormesh(*args, **kwargs)
    c_max = np.max(args[2])
    c_min = np.min(args[2])
#TODO try heatmap coloring viridis

def plot(*args, **kwarg):
    global colorscheme
    global colorcounter
    if(not kwarg):
        kwarg= {'color':getColor()}

    if('color' in kwarg or 'c' in kwarg and len(args) < 3):
        p = plt.plot(*args, **kwarg)[0]
    elif('color' in kwarg or 'c' in kwarg):
        p = plt.plot(*args, **kwarg)[0]
    elif(len(args) < 3):
        p = plt.plot(*args, **dict(kwarg, color=getColor()))[0]
    else:
        p = plt.plot(*args, **dict(kwarg, color=getColor()))[0]
    lines.append(p)

def xlim(x,y):
    ax.set_xlim([x,y])

def ylim(x,y):
    ax.set_ylim([x,y])

def zlim(x,y):
    ax.set_zlim([x,y])

def clim(min_val,max_val):
    global cbar
    cbar.set_clim([min_val,max_val])

    cbar.set_ticks(getTickSpacing(min_val, max_val))



    cbar.update_ticks()

def show():
    if plotstyle == 'descriptive':
        drawArrows()
    fig.tight_layout()
    fig.show()


def setXticks(ticks):
    ax.set_xticks(ticks)

def setXlabels(labels):
    ax.set_xticklabels(labels)

def setYticks(ticks):
    ax.set_yticks(ticks)

def setYlabels(labels):
    ax.set_yticklabels(labels)

def savefig(filepath):
    if plotstyle == 'descriptive':
        drawArrows()
    if legendartist:
            fig.savefig(filepath, bbox_extra_artists=(legendartist,), bbox_inches='tight')
    else:
        fig.tight_layout()
        fig.savefig(filepath)

def text(*args, **kwargs):
    if not kwargs:
        kwargs = dict(color=almost_black)
    if not 'color' in kwargs:
        kwargs = dict(kwargs, color=almost_black)
    plt.text(*args, **kwargs)


def close():
    global colorcounter
    plt.close()
    colorcounter=0


def drawArrows():
    for tick in ax.xaxis.get_majorticklabels():
        tick.set_horizontalalignment("left")
    for tick in ax.yaxis.get_majorticklabels():
        tick.set_horizontalalignment("left")
        tick.set_verticalalignment("bottom")

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    dps = fig.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(dps)
    width, height = bbox.width, bbox.height

    # manual arrowhead width and length
    hw = 1./50.*(ymax-ymin)
    hl = 1./50.*(xmax-xmin)
    lw = 1.0 # axis line width
    ohg = 0.1 # arrow overhang

    # compute matching arrowhead length and width
    yhw = hw/(ymax-ymin)*(xmax-xmin)* height/width
    yhl = hl/(xmax-xmin)*(ymax-ymin)* width/height

    # draw x and y axis
    ax.arrow(xmin, 0, xmax-xmin, 0., fc='k', ec='k', lw = lw,
             head_width=hw, head_length=hl, overhang = ohg,
             length_includes_head= True, clip_on = False, color=almost_black)

    ax.arrow(0, ymin, 0., ymax-ymin, fc='k', ec='k', lw = lw,
             head_width=yhw, head_length=yhl, overhang = ohg,
             length_includes_head= True, clip_on = False, color=almost_black)

def addLegendText(text):
    legend_texts.append(text)

def legend(**kwargs):
    global legendartist
    if not kwargs:
        kwargs = dict(loc=1)
    if kwargs.pop('outside', False):
        if(cbar):
             kwargs=dict(kwargs, bbox_to_anchor=(1.61, 1.05))
        else:
            kwargs=dict(kwargs, bbox_to_anchor=(0.2, 1.05))
    if 'frameon' not in kwargs:
        kwargs=dict(kwargs,  frameon=False)
    if(legend_texts):
        legendartist = ax.legend(lines, legend_texts, **kwargs)
    else:
        legendartist = ax.legend(**kwargs)

def arrow(*args, **kwargs):
    if not kwargs:
        kwargs = dict(color=getColor())
    if not 'color' in kwargs:
        kwargs = dict(kwargs, color=getColor())
    if not 'mutation_scale' in kwargs:
        kwargs = dict(kwargs, mutation_scale=20)
    if not 'arrowstyle' in kwargs:
        kwargs = dict(kwargs, arrowstyle="->")
    if not 'lw' in kwargs:
        kwargs = dict(kwargs, lw=linewidth)
    arrow = Arrow3D(*args, **kwargs)
    ax.add_artist(arrow)

def getColor():
    global colorcounter
    color = tableau20[colorcounter]
    if(useHigherColorContrast):
        colorcounter+=2
    else:
        colorcounter+=1
    if colorcounter >= len(tableau20):
        colorcounter=0
    return color

def setColorBack():
    global colorcounter
    if(useHigherColorContrast):
        colorcounter-=2
    else:
        colorcounter-=1


def plane3D(point, normal_vector, lim=1.0, **kwargs):
    if not kwargs:
        kwargs = dict(color=getColor())
    if not 'color' in kwargs:
        kwargs = dict(kwargs, color=getColor())
    # a plane is a*x+b*y+c*z+d=0
    # [a,b,c] is the normal. Thus, we have to calculate
    # d and we're set
    point = np.array(point)
    normal_vector = np.array(normal_vector)
    d = -point.dot(normal_vector)

    # create x,y

    nr_points=101
    xx, yy = np.meshgrid(np.linspace(point[0]-lim, point[0]+lim, nr_points),
                         np.linspace(point[1]-lim, point[1]+lim, nr_points))

    # calculate corresponding z
    if (normal_vector[2]==0.0):
        normal_vector[2] = 1E-99
    z = (-normal_vector[0] * xx - normal_vector[1] * yy - d) * 1. /normal_vector[2]

    # plot the surface
    surf = ax.plot_surface(xx, yy, z)
    surf.set_alpha(.1)

def view_init(elev, azimut):
    ax.view_init(elev, azimut)

def plot_wireframe(*args, **kwargs):
    global colorcounter
    if not kwargs:
        kwargs= {'color':getColor()}
    if 'color' in kwargs:
        ax.plot_wireframe(*args, **kwargs)
    else:
        ax.plot_wireframe(*args, **dict(kwargs, color=getColor()))


#TODO make a aspect ratio function for heatmaps of xy-spatial plots


def rectangle(*args,**kwargs):
    currentAxis = plt.gca()
    if not kwargs:
        kwargs= {'color':getColor()}
    currentAxis.add_patch(Rectangle(*args, **kwargs))

def hline(y, **kwargs):
    if not kwargs:
        kwargs= {'color':almost_black}
    if 'color' not in kwargs:
        kwargs = dict(kwargs, color=almost_black)
    if 'lw' not in kwargs:
        lw = ax.spines['bottom'].get_linewidth()
        kwargs = dict(kwargs, lw=lw)
    lims = ax.get_xlim()
    plot([lims[0],lims[1]], [y,y], **kwargs)

def vline(x, **kwargs):
    if not kwargs:
        kwargs= {'color':almost_black}
    if 'color' not in kwargs:
        kwargs = dict(kwargs, color=almost_black)
    if 'lw' not in kwargs:
        lw = ax.spines['left'].get_linewidth()
        kwargs = dict(kwargs, lw=lw)
    lims = ax.get_ylim()
    plot([x,x], [lims[0],lims[1]],  **kwargs)

def fill_between(*args, **kwargs):
    if not kwargs:
        kwargs= {'facecolor':red}
    if 'facecolor' not in kwargs:
        kwargs = dict(kwargs, facecolor=red)
    if 'alpha' not in kwargs:
        kwargs = dict(kwargs, alpha=0.5)

    ax.fill_between(*args, **kwargs)