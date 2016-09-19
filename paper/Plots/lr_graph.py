import numpy as np  
import matplotlib.pyplot as plt

def graph(formula, x_range, fname):  
    x = np.array(x_range)  
    y = eval(formula)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y)  
    ax.set_xlabel('Iteration Count', fontsize=10)
    ax.set_ylabel('Learning Rate', fontsize=10)
    fig.savefig(fname + ".png")
    # plt.show()

graph('0.001 * np.power((1 + 0.0001 * x),(-0.75))', range(0,1000), "lr_inv")

def graph_lr_step(formula, x_range):  
	f, (ax, ax2) = plt.subplots(2, 1, sharex=True)

	x = np.array(x_range)  
	y = eval(formula)
	# plot the same data on both axes
	ax.plot(x,y)
	ax2.plot(x,y)

	# zoom-in / limit the view to different portions of the data
	ax.set_ylim(0.0006, 0.0012)  # outliers only
	ax2.set_ylim(0, .0002)  # most of the data

	# hide the spines between ax and ax2
	ax.spines['bottom'].set_visible(False)
	ax2.spines['top'].set_visible(False)
	ax.xaxis.tick_top()
	ax.tick_params(labeltop='off')  # don't put tick labels at the top
	ax2.xaxis.tick_bottom()
	ax2.set_xlabel('Iteration Count', fontsize=10)
	f.text(0.015, 0.5, 'Learning Rate', fontsize=10, va='center', rotation='vertical')
	# ax2.set_ylabel(, fontsize=10)

	# This looks pretty good, and was fairly painless, but you can get that
	# cut-out diagonal lines look with just a bit more work. The important
	# thing to know here is that in axes coordinates, which are always
	# between 0-1, spine endpoints are at these locations (0,0), (0,1),
	# (1,0), and (1,1).  Thus, we just need to put the diagonals in the
	# appropriate corners of each of our axes, and so long as we use the
	# right transform and disable clipping.

	d = .015  # how big to make the diagonal lines in axes coordinates
	# arguments to pass plot, just so we don't keep repeating them
	kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
	ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
	ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

	kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
	ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
	ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

	# What's cool about this is that now if we vary the distance between
	# ax and ax2 via f.subplots_adjust(hspace=...) or plt.subplot_tool(),
	# the diagonal lines will move accordingly, and stay right at the tips
	# of the spines they are 'breaking'

	plt.savefig("lr_step.png")

graph_lr_step('0.001 * np.power(0.1,(np.floor(x/5000)))',range(0,20000))