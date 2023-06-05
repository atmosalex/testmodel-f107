import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from math import log10, sqrt, ceil
import os
import sys
import colorsys

def get_N_cols(N):
    HSV_tuples = [(x * 1.0 / N, 0.74, 0.7) for x in range(N)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    return RGB_tuples


def plot_partitioned_data(dir_plots, time, observed, idx_part1, idx_part2, idx_part1_gap, target_key,
                          plot_logy=True, subdir="splittesttrain", yrange_ordersofmag=4.5, generic_title=False,
                          tlimits=[], ylimits=[], colour_custom=None, colour_dict=None):
    dir_output = os.path.join(dir_plots, subdir)
    print("", f"saving {target_key} plots to", dir_output)

    # plot the forecast across all the testing data provided
    # the data will be split onto different plots with a maximum time window of dt_maxplot
    dt_maxplot = pd.Timedelta(60, "days")
    dt_1unit = pd.Timedelta(1, "hour")

    fontsize = 8.5

    # the plot will be split over multiple time windows depending on dt_maxplot:
    if len(tlimits):
        tmax = tlimits[1]
        tmin = tlimits[0]
    else:
        tmax = time.max()
        tmin = time.min()
    ntwin = ceil((tmax - tmin).total_seconds() / dt_maxplot.total_seconds())

    # find the y limits to apply to every subplot for consistency when comparing:
    if len(ylimits):
        ymax_all = ylimits[1]
        ymin_all = ylimits[0]
    else:
        q99, q1 = np.percentile(observed, [99, 1])
        ymax_all = q99
        ymin_all = q1

    # n_rearrangements = len(idx_part1_gap)
    # for idx_split_tv_test in range(n_rearrangements):
    fig, axs = plt.subplots(ntwin, figsize=(8, 1 + 1.8 * ntwin), sharex=False)
    if ntwin == 1: axs = [axs]

    axs[0].set_title(f"{target_key}", fontdict={'fontsize': fontsize + 2})

    if colour_custom is not None:
        col = colour_custom
        col_dict = colour_dict
    else:
        col = np.array(['r'] * len(time))
        col[idx_part2] = 'b'
        col_dict = {'r': 'part 1', 'b': 'part 2'}

    time_part1 = time[idx_part1]
    time_part2 = time[idx_part2]
    observed_part1 = observed[idx_part1]
    observed_part2 = observed[idx_part2]
    col_part1 = [col[idx] for idx in idx_part1]
    col_part2 = [col[idx] for idx in idx_part2]

    time_maxplot_prev = tmin
    for itwin in range(ntwin):
        ax = axs[itwin]
        ax.set_axisbelow(True)

        time_minplot = time_maxplot_prev
        time_maxplot = time_minplot + dt_maxplot
        fill = -1

        # plot training data inside this window
        time_part1idx = range(len(time_part1))
        time_part1idxinrange = np.where(np.logical_and(time_part1 >= time_minplot, time_part1 <= time_maxplot),
                                        time_part1idx, fill)
        time_part1idxinrange = time_part1idxinrange[time_part1idxinrange != fill].astype(int)
        ax.scatter(time_part1[time_part1idxinrange], observed_part1[time_part1idxinrange], marker='.',
                   color=[col_part1[idx] for idx in time_part1idxinrange], s=0.2, zorder=1)
        time_part1idxinrange_0 = time_part1idxinrange[time_part1idxinrange < idx_part1_gap]
        time_part1idxinrange_1 = time_part1idxinrange[time_part1idxinrange >= idx_part1_gap]
        ax.plot(time_part1[time_part1idxinrange_0], observed_part1[time_part1idxinrange_0], color='black', lw=0.2,
                zorder=2)
        ax.plot(time_part1[time_part1idxinrange_1], observed_part1[time_part1idxinrange_1], color='black', lw=0.2,
                zorder=2)

        # plot testing data inside this window
        time_part2idx = range(len(time_part2))
        time_part2idxinrange = np.where(np.logical_and(time_part2 >= time_minplot, time_part2 <= time_maxplot),
                                        time_part2idx, fill)
        time_part2idxinrange = time_part2idxinrange[time_part2idxinrange != fill].astype(int)
        ax.scatter(time_part2[time_part2idxinrange], observed_part2[time_part2idxinrange], marker='.',
                   color=[col_part2[idx] for idx in time_part2idxinrange], s=0.2, zorder=2)
        time_part2idxinrange_0 = time_part2idxinrange[time_part2idxinrange < idx_part1_gap]
        time_part2idxinrange_1 = time_part2idxinrange[time_part2idxinrange >= idx_part1_gap]
        ax.plot(time_part2[time_part2idxinrange_0], observed_part2[time_part2idxinrange_0], color='black', lw=0.2,
                zorder=2)
        ax.plot(time_part2[time_part2idxinrange_1], observed_part2[time_part2idxinrange_1], color='black', lw=0.2,
                zorder=2)

        ax.grid(which='both')

        # t axis:
        ax.set_xlim([time_minplot, time_maxplot])
        time_maxplot_prev = time_maxplot  # for the next plotting window
        for label in ax.get_xticklabels():
            label.set_fontsize(fontsize - 3)
            label.set_ha('right')
            label.set_rotation(0.)

        # y axis:
        for label in ax.get_yticklabels():
            label.set_fontsize(fontsize)
        if plot_logy:
            ax.set_yscale('log')
            ymin = max(ymin_all, 10 ** (log10(ymax_all) - yrange_ordersofmag))
        else:
            ymin = ymin_all
        ax.set_ylim([ymin, ymax_all])
        ax.set_ylabel('y')

    for c in col_dict.keys():
        ax.scatter([time[0]], [ymin / 2], label=col_dict[c], color=c)
    ax.legend()

    dir_output = os.path.join(dir_plots, subdir)
    if not os.path.isdir(dir_output):
        try:
            os.mkdir(dir_output)
        except:
            print(f"could not make directory {dir_output}")
            sys.exit()
    fname = f'{target_key}.png'
    path = os.path.join(dir_plots, subdir, fname)

    plt.savefig(path, bbox_inches='tight', dpi=250)
    plt.close()

    return [ymin_all, ymax_all]


def plot_rawdata(dir_plots, time, y, key, plot_logy, subdir="", yrange_ordersofmag=4.5, tlimits=[], ylimits=[],
                 colour_custom=None, colour_dict=None):
    # a wrapper for plot_partitioned_data, but in this case the data is not partitioned, so pretend there exists another partition of size 0:
    ylimits = plot_partitioned_data(dir_plots, time, y, [0], np.arange(len(y)), [0], key,
                                    plot_logy=plot_logy, subdir=subdir, yrange_ordersofmag=yrange_ordersofmag,
                                    generic_title=True,
                                    tlimits=tlimits, ylimits=ylimits, colour_custom=colour_custom,
                                    colour_dict=colour_dict)
    return ylimits


def plot_forecast(dir_plots, time, observed, forecasted, persistence, info_model, plot_logy=True, subdir="test",
                  yrange_ordersofmag=4.5, version=None, ntwin_max=10):
    dir_output = os.path.join(dir_plots, subdir)
    print("#", "saving forecast plots to", dir_output)

    # plot the forecast across all the testing data provided
    # the data will be split onto different plots with a maximum time window of dt_maxplot
    dt_maxplot = pd.Timedelta(365, "days")  # shorter time windows are useful for identifying lag
    dt_1unit = pd.Timedelta(1, "day")

    fh_max = forecasted.shape[0]
    target_key = info_model['target_key']
    size_batch = info_model['size_batch']
    seqlen = info_model['seqlen']
    learning_rate = info_model['learning_rate_start']
    n_hidden = info_model['n_hidden']
    scaling_method = info_model['scaling_method']
    fh_RMSE = info_model['fh_RMSE']
    fh_RMSE_persistence = info_model['fh_RMSE_persistence']
    fontsize = 8.5

    # the plot will be split over multiple time windows depending on dt_maxplot:
    ntwin = ceil((time[-1] - time[0]).total_seconds() / dt_maxplot.total_seconds())
    ntwin = min(ntwin_max, ntwin)
    # first, find the y limits to apply to every subplot for consistency when comparing:
    ymin_allfh = np.ones(fh_max) * -np.inf
    ymax_allfh = np.ones(fh_max) * np.inf
    for fh in range(fh_max):
        ymin_pc = 1
        ymax_pc = 99
        ymin = min([np.min(forecasted[fh, :]),
                    np.min(observed),
                    np.min(persistence[fh, :])])
        ymax = max([np.max(forecasted[fh, :]),
                    np.max(observed),
                    np.max(persistence[fh, :])])
        ymin = min([np.percentile(forecasted[fh, :], [ymin_pc]),
                    np.percentile(observed, [ymin_pc]),
                    np.percentile(persistence[fh, :], [ymin_pc])])
        ymax = max([np.percentile(forecasted[fh, :], [ymax_pc]),
                    np.percentile(observed, [ymax_pc]),
                    np.percentile(persistence[fh, :], [ymax_pc])])
        ymin_allfh[fh] = ymin
        ymax_allfh[fh] = ymax

    for fh in range(fh_max):
        fig, axs = plt.subplots(ntwin, figsize=(8, 1 + 2 * ntwin), sharex=False)
        if ntwin == 1: axs = [axs]

        axs[0].set_title(f"{target_key} {fh + 1} Day Forecast", fontdict={'fontsize': fontsize + 2})

        # fig.autofmt_xdate() #deletes axis labels on last row

        time_maxplot_prev = time[0]
        for itwin in range(ntwin):
            ax = axs[itwin]
            ax.set_axisbelow(True)

            time_minplot = time_maxplot_prev
            time_maxplot = time_minplot + dt_maxplot
            fill = -1
            timeidx = range(len(time))
            timeidxinrange = np.where(np.logical_and(time >= time_minplot, time <= time_maxplot), timeidx, fill)
            timeidxinrange = timeidxinrange[timeidxinrange != fill].astype(int)

            # plot persistence
            ax.plot((time + fh * dt_1unit), persistence[fh, :], 'grey', linewidth=0.8, alpha=0.8, label='persistence',
                    zorder=1)
            # plot data
            # ax.scatter(time, observed, marker='.', color='black', s=1, label='data', zorder = 2)
            ax.plot(time, observed, 'black', linewidth=0.8, alpha=0.7, label='data', zorder=2)
            # plot model
            ax.plot((time + fh * dt_1unit), forecasted[fh, :], 'blue', linewidth=0.8, alpha=0.7, label='model', zorder=3)

            ax.grid(which='both')

            # x axis:
            ax.set_xlim([time_minplot, time_maxplot])
            time_maxplot_prev = time_maxplot  # for the next plotting window
            for label in ax.get_xticklabels():
                label.set_ha('right')
                label.set_rotation(0.)
                label.fontsize = fontsize - 3

            # y axis:
            # ylims = ax.get_ylim()
            for label in ax.get_yticklabels():
                label.fontsize = fontsize
            ymax = max(ymax_allfh)
            if plot_logy:
                ax.set_yscale('log')
                ymin = max(min(ymin_allfh), 10 ** (log10(ymax) - yrange_ordersofmag))
            else:
                ymin = min(ymin_allfh)

            ax.set_ylim([ymin, ymax])
            ax.set_ylabel('y')

        axs[-1].legend(loc='lower right', fontsize=fontsize,
                       facecolor='white', edgecolor='grey', framealpha=0.6, borderpad=0.1)

        # print hyperparameters
        hyperparams = f'batch size= {size_batch}\n' \
                      f'seq. len  = {seqlen}\n' \
                      f'learn rate= {learning_rate:.3E}\n' \
                      f'n. hidden = {n_hidden}\n' \
                      f'scal. method = {scaling_method}'
        axs[-1].text(0.01, 0.02, hyperparams, ha='left', va='bottom', transform=ax.transAxes,
                     fontdict={'fontsize': fontsize, 'family': 'monospace'},
                     bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round', alpha=0.6, pad=0.1))
        # print score
        score = 'error in f\n' \
                f'RMSE = {fh_RMSE[fh]:.4f}\n' \
                f'RMSEp= {fh_RMSE_persistence[fh]:.4f}'
        axs[-1].text(0.5, 0.02, score, ha='center', va='bottom', transform=ax.transAxes,
                     fontdict={'fontsize': fontsize, 'family': 'monospace'},
                     bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round', alpha=0.6, pad=0.1))

        dir_output = os.path.join(dir_plots, subdir)
        if not os.path.isdir(dir_output):
            try:
                os.mkdir(dir_output)
            except:
                print(f"could not make directory {dir_output}")
                sys.exit(1)

        if version is not None:
            fname = f'predict_fh{fh + 1}_tvtest{version}.png'
        else:
            fname = f'predict_fh{fh + 1}.png'
        path = os.path.join(dir_plots, subdir, fname)
        plt.savefig(path, bbox_inches='tight', dpi=250)
        plt.close()



