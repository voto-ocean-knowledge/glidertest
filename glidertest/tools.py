import numpy as np
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns
from scipy import stats
from cmocean import cm as cmo
import matplotlib.pyplot as plt


def grid2d(x, y, v, xi=1, yi=1):
    if np.size(xi) == 1:
        xi = np.arange(np.nanmin(x), np.nanmax(x) + xi, xi)
    if np.size(yi) == 1:
        yi = np.arange(np.nanmin(y), np.nanmax(y) + yi, yi)
    raw = pd.DataFrame({'x': x, 'y': y, 'v': v}).dropna()
    grid = np.full([np.size(xi), np.size(yi)], np.nan)
    raw['xbins'], xbin_iter = pd.cut(raw.x, xi, retbins=True, labels=False)
    raw['ybins'], ybin_iter = pd.cut(raw.y, yi, retbins=True, labels=False)
    _tmp = raw.groupby(['xbins', 'ybins'])['v'].agg('median')
    grid[_tmp.index.get_level_values(0).astype(int), _tmp.index.get_level_values(1).astype(int)] = _tmp.values
    YI, XI = np.meshgrid(yi, xi)
    return grid, XI, YI


def test_updown_bias(ds, axis, var='PSAL', v_res=0, return_val=False):
    p = 1  # Horizontal resolution
    z = v_res  # Vertical resolution
    varG, profG, depthG = grid2d(ds.PROFILE_NUMBER, ds.DEPTH, ds[var], p, z)

    grad = np.diff(varG, axis=0)  # Horizontal gradients

    dc = np.nanmean(grad[0::2, :], axis=0)  # Dive - CLimb
    cd = np.nanmean(grad[1::2, :], axis=0)  # Climb - Dive
    axis.plot(dc, depthG[0, :], label='Dive-Climb')
    axis.plot(cd, depthG[0, :], label='Climb-Dive')
    axis.legend(loc=3)
    lims = np.abs(dc)
    axis.set_xlim(-np.nanpercentile(lims, 99.5), np.nanpercentile(lims, 99.5))
    axis.set_xlabel(ds[var].attrs['long_name'])
    axis.set_ylim(depthG.max() + 10, -5)
    axis.grid()
    if return_val:
        return dc, cd


def find_cline(var, depth_array):
    clin = np.where(np.abs(np.diff(np.nanmean(var, axis=0))) == np.nanmax(np.abs(np.diff(np.nanmean(var, axis=0)))))
    return np.round( depth_array[0, clin[0]], 1)


def plot_basic_vars(ds, v_res=1, start_prof=0, end_prof=-1):
    p = 1
    z = v_res
    tempG, profG, depthG = grid2d(ds.PROFILE_NUMBER, ds.DEPTH, ds.TEMP, p, z)
    salG, profG, depthG = grid2d(ds.PROFILE_NUMBER, ds.DEPTH, ds.PSAL, p, z)
    denG, profG, depthG = grid2d(ds.PROFILE_NUMBER, ds.DEPTH, ds.DENSITY, p, z)

    tempG = tempG[start_prof:end_prof, :]
    salG = salG[start_prof:end_prof, :]
    denG = denG[start_prof:end_prof, :]
    depthG = depthG[start_prof:end_prof, :]

    halo = find_cline(salG, depthG)
    thermo = find_cline(tempG, depthG)
    pycno = find_cline(denG, depthG)
    print(
        f'The thermocline, halocline and pycnocline are located at respectively {thermo}, {halo} and {pycno}m as shown in the plots as well')

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax1 = ax[0].twiny()
    ax2 = ax[0].twiny()
    ax2.spines["top"].set_position(("axes", 1.2))
    ax[0].plot(np.nanmean(tempG, axis=0), depthG[0, :], c='blue')
    ax1.plot(np.nanmean(salG, axis=0), depthG[0, :], c='red')
    ax2.plot(np.nanmean(denG, axis=0), depthG[0, :], c='black')
    ax[0].axhline(thermo, linestyle='dashed', c='blue')
    ax1.axhline(halo, linestyle='dashed', c='red')
    ax2.axhline(pycno, linestyle='dashed', c='black')

    ax[0].set(xlabel=f'Average Temperature [C] \nbetween profile {start_prof} and {end_prof}', ylabel='Depth (m)')
    ax[0].tick_params(axis='x', colors='blue')
    ax[0].xaxis.label.set_color('blue')
    ax1.spines['bottom'].set_color('blue')
    ax1.set(xlabel=f'Average Salinity [PSU] \nbetween profile {start_prof} and {end_prof}')
    ax1.xaxis.label.set_color('red')
    ax1.spines['top'].set_color('red')
    ax1.tick_params(axis='x', colors='red')
    ax2.spines['bottom'].set_color('black')
    ax2.set(xlabel=f'Average Density [kg m-3] \nbetween profile {start_prof} and {end_prof}')
    ax2.xaxis.label.set_color('black')
    ax2.spines['top'].set_color('black')
    ax2.tick_params(axis='x', colors='black')

    if 'CHLA' in ds.variables:
        chlaG, profG, depthG = grid2d(ds.PROFILE_NUMBER, ds.DEPTH, ds.CHLA, p, z)
        chlaG = chlaG[start_prof:end_prof, :]
        ax2_1 = ax[1].twiny()
        ax2_1.plot(np.nanmean(chlaG, axis=0), depthG[0, :], c='green')
        ax2_1.set(xlabel=f'Average Chlorophyll-a [mg m-3] \nbetween profile {start_prof} and {end_prof}')
        ax2_1.xaxis.label.set_color('green')
        ax2_1.spines['top'].set_color('green')
        ax2_1.tick_params(axis='x', colors='green')
    else:
        ax[1].text(0.3, 0.7, 'Chlorophyll data unavailable', va='top', transform=ax[1].transAxes)

    if 'DOXY' in ds.variables:
        oxyG, profG, depthG = grid2d(ds.PROFILE_NUMBER, ds.DEPTH, ds.DOXY, p, z)
        oxyG = oxyG[start_prof:end_prof, :]
        ax[1].plot(np.nanmean(oxyG, axis=0), depthG[0, :], c='orange')
        ax[1].set(xlabel=f'Average Oxygen [mmol m-3] \nbetween profile {start_prof} and {end_prof}')
        ax[1].xaxis.label.set_color('orange')
        ax[1].spines['top'].set_color('orange')
        ax[1].tick_params(axis='x', colors='orange')
        ax[1].spines['bottom'].set_color('orange')
    else:
        ax[1].text(0.3, 0.5, 'Oxygen data unavailable', va='top', transform=ax[1].transAxes)

    [a.set_ylim(depthG.max() + 10, -5) for a in ax]
    [a.grid() for a in ax]


# Check if there is any negative scaled data and/or raw data
def chl_first_check(ds):
    # Check how much negative data there is
    neg_chl = np.round((len(np.where(ds.CHLA < 0)[0]) * 100) / len(ds.CHLA), 1)
    if neg_chl > 0:
        print(f'{neg_chl}% of scaled chlorophyll data is negative, consider recalibrating data')
        # Check where the negative values occur and if we just see them at specific time of the mission or not
        start = ds.TIME[np.where(ds.CHLA < 0)][0]
        end = ds.TIME[np.where(ds.CHLA < 0)][-1]
        min_z = (np.round(ds.DEPTH[np.where(ds.CHLA < 0)].min().values, 1))
        max_z = (np.round(ds.DEPTH[np.where(ds.CHLA < 0)].max().values, 1))
        print(f'Negative data in present from {str(start.values)[:16]} to {str(end.values)[:16]}')
        print(f'Negative data is present between {min_z} and {max_z} ')
    else:
        print('There is no negative scaled chlorophyll data, recalibration and further checks are still recommended ')
    # Check if there is any missing data throughout the mission
    if len(ds.TIME) != len(ds.CHLA.dropna(dim='N_MEASUREMENTS').TIME):
        print('Chlorophyll data is missing for part of the mission')  # Add to specify where the gaps are
    else:
        print('Chlorophyll data is present for the entire mission duration')
    # Check bottom dark count and any drift there
    bottom_chl_data = ds.CHLA.where(ds.CHLA.DEPTH > ds.DEPTH.max() - (ds.DEPTH.max() * 0.1)).dropna(
        dim='N_MEASUREMENTS')
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(0, len(bottom_chl_data)), bottom_chl_data)
    ax = sns.regplot(data=ds, x=np.arange(0, len(bottom_chl_data)), y=bottom_chl_data,
                     scatter_kws={"color": "grey"},
                     line_kws={"color": "red", "label": "y={0:.6f}x+{1:.3f}".format(slope, intercept)},
                     )
    ax.legend(loc=2)
    ax.grid()
    ax.set(ylim=(np.nanpercentile(bottom_chl_data, 0.1), np.nanpercentile(bottom_chl_data, 99.9)),
           xlabel='Measurements',
           ylabel='Chla')
    plt.show()
    if slope >= 0.00001:
        print(
            'Data from the deepest 10% of data has been analysed and data does not seem stable. An alternative solution for dark counts has to be considered. \nMoreover, it is recommended to check the sensor has this may suggest issues with the sensor (i.e water inside the sensor, temporal drift etc)')
    else:
        print(
            'Data from the deepest 10% of data has been analysed and data seems stable. These deep values can be used to re-assess the dark count if the no chlorophyll at depth assumption is valid in this site and this depth')
