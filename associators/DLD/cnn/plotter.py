import matplotlib as mpl
from matplotlib import pyplot as plt
mpl.use("Agg")


def plot_hist(ap_dists, an_dists, nbins, axs=None):
    import numpy as np
    if not axs:
        plt.figure()
        axs = plt.axes()

    pos_masked = np.ma.masked_array(ap_dists, mask=ap_dists < 0)
    neg_masked = np.ma.masked_array(an_dists, mask=an_dists < 0)
    rang0 = neg_masked.max()

    phist = _plot_hist_helper(axs, pos_masked, rang0, nbins, "g")
    nhist = _plot_hist_helper(axs, neg_masked, rang0, nbins, "r")

    return phist, nhist


def _plot_hist_helper(ax, els, rang, bins, color):
    import numpy as np
    from matplotlib.colors import to_rgba
    ma = np.ma.masked_array(els, mask=els < 0)

    n, nbins, _ = ax.hist(ma, bins=bins, range=(0, rang),
                          color=to_rgba(color, 0.5))

    # scale to data
    r = np.argmax(n != 0)
    r2 = len(n) - np.argmax(n[::-1] != 0)

    ax.axvspan(nbins[r], nbins[r2], facecolor=color, alpha=0.1)

    return n, nbins


def get_roc_data(ap_dists, an_dists, nbins):
    import numpy as np
    # example from https://towardsdatascience.com/receiver-operating-characteristic-curves-demystified-in-python-bd531a4364d0

    pos_masked = np.ma.masked_array(ap_dists, mask=ap_dists < 0)
    neg_masked = np.ma.masked_array(an_dists, mask=an_dists < 0)
    rang0 = [min(pos_masked.min(),neg_masked.min()),
             max(neg_masked.max(), pos_masked.max())]
    # create bins for cnn
    bins = np.linspace(rang0[0], rang0[1], num=nbins + 1)

    h0 = np.histogram(pos_masked, range=rang0, bins=bins)
    h1 = np.histogram(neg_masked, range=rang0, bins=bins)
    tpos = h0[0].copy()
    tneg = h1[0].copy()

    # Total
    pn = np.sum(tpos)
    nn = np.sum(tneg)

    # Cumulative sum
    cum_TP = 0
    cum_FP = 0

    # TPR and FPR list initialization
    TPR_list = [0]
    FPR_list = [0]

    # Iteratre through all values of x
    for i in range(len(bins) - 1):
        # We are only interested in non-zero values of bad
        cum_TP += tpos[i]
        cum_FP += tneg[i]
        FPR = cum_FP / nn
        TPR = cum_TP / pn
        TPR_list.append(TPR)
        FPR_list.append(FPR)

    TPR_list.append(1)
    FPR_list.append(1)
    TPR_list = np.array(TPR_list)
    FPR_list = np.array(FPR_list)

    return TPR_list, FPR_list


def _plot_roc_helper(axs, ap_dists, an_dists, nbins, label, col):
    import matplotlib.pyplot # noqa
    import numpy as np

    blank = mpl.patches.Rectangle((0, 0), 0.1, 0.1, fc="w", fill=False,
                                  edgecolor='none', linewidth=0)

    TPR_list, FPR_list = get_roc_data(ap_dists, an_dists, nbins)
    # Plotting final ROC curve
    x95 = np.interp(0.95, TPR_list, FPR_list)
    x95p = x95 * 100

    for ax in axs:
        # for axs in [ax, ax2]:
        ax.plot(FPR_list, TPR_list,
                color=col, label="{} |= {:.3f}%".format(label, x95p))

        # ax.set_xlim([-0.1,1.1])
        # ax.set_ylim([-0.1,1.1])
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_title("ROC Curve", fontsize=14)
        ax.set_ylabel('TPR', fontsize=12)
        ax.set_xlabel('FPR', fontsize=12)
        ax.grid(True)

        ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
        ax.yaxis.set_minor_formatter(mpl.ticker.PercentFormatter(1))
        ax.yaxis.set_minor_locator(mpl.ticker.FixedLocator([0.95]))

        # sort legends
        plots, legends = ax.get_legend_handles_labels()
        legends = [el.split("|") for el in legends]

        def sorting_key(el):
            return float(legends[el[0]][1].split("= ")[1][:-1])

        plots = sorted([[i, el, blank] for i, el in enumerate(plots)],
                       key=sorting_key)
        plots = [el[1:] for el in plots]
        plots = np.array(plots).T.flatten()

        def sorting_key(el):
            return float(el[1].split("= ")[1][:-1])

        legends = sorted(legends, key=sorting_key)
        legends = np.array(legends).T.flatten()

        # limit
        ax.legend(plots, legends, columnspacing=-2.5, ncol=2)

    axs[1].set_xlim([-0.002, min(x95 * 2, FPR_list[-1])])
    axs[1].set_ylim([0.90, 1.002])

    return FPR_list, TPR_list


def plot_roc(ap_dists, an_dists, label, nbins=200,
             color="g", figs=None):
    if not figs:
        figs = []
        for _ in range(2):
            fig = plt.figure(figsize=(10.24, 7.68))
            ax = plt.axes()
            figs.append(fig)  # 1024 x 768

        # axs = plt.axes()
    axs = [fig.axes[0] for fig in figs]
    fp0, tp0 = _plot_roc_helper(axs, ap_dists, an_dists,
                                nbins, label,
                                color)
    for ax in axs:
        ax.plot([-1, 1], [0.95, 0.95], "--", color=(0.3, 0.3, 0.3, 0.5))
        ax.plot([-1, 1], [0.95, 0.95], "--", color=(0.3, 0.3, 0.3, 0.5))
        ax.plot([-1, 1], [-1, 1], "--", color=(0.3, 0.3, 0.3, 0.5))
        ax.plot([-1, 1], [-1, 1], "--", color=(0.3, 0.3, 0.3, 0.5))

    return figs


def plot_to_np(fig):
    import numpy as np
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = h, w, 3
    return buf
