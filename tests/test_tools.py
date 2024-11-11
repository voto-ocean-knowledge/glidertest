import pytest
from glidertest import fetchers, tools
import matplotlib.pyplot as plt
import math


def test_plots():
    ds = fetchers.load_sample_dataset()
    tools.plot_basic_vars(ds,start_prof=0, end_prof=int(ds.PROFILE_NUMBER.max()))
    assert ax[0].get_ylabel() == 'Depth (m)'
    assert ax[0].get_xlabel() == f'Average Temperature [C] \nbetween profile {start_prof} and {end_prof}'
    return fig


def test_up_down_bias():
    ds = fetchers.load_sample_dataset()
    fig, ax = plt.subplots()
    df = tools.updown_bias(ds, var='PSAL', v_res=1)
    bins = np.unique(np.round(ds.DEPTH,0))
    ncell = math.ceil(len(bins)/v_res)
    assert len(df) == ncell
    tools.plot_updown_bias(df, ax,  xlabel='Salinity')
    lims = np.abs(df.dc)
    assert ax.get_xlim() == (-np.nanpercentile(lims, 99.5), np.nanpercentile(lims, 99.5))
    assert ax.get_ylim() == (df.depth.max() + 1, -df.depth.max() / 30)
    assert ax[0].get_xlabel() == xlabel
 

def test_chl():
    ds = fetchers.load_sample_dataset()
    tools.optics_first_check(ds, var='CHLA')
    assert ax.get_ylabel() == var
    tools.optics_first_check(ds, var='BBP700')
    assert ax.get_ylabel() == var
    with pytest.raises(KeyError) as e:
        tools.optics_first_check(ds, var='nonexistent_variable')


def test_quench_sequence():
    ds = fetchers.load_sample_dataset()
    if not "TIME" in ds.indexes.keys():
        ds = ds.set_xindex('TIME')
    fig, ax = plt.subplots()
    tools.plot_section_with_srss(ds, 'CHLA')
    assert ax.get_ylabel == 'Depth [m]'
    assert ax.get_ylim() == (ylim, -ylim / 30)
    
    dayT, nightT = tools.day_night_avg(ds, sel_var='TEMP')
    assert len(nightT.dat.dropna()) > 0
    assert len(dayT.dat.dropna()) > 0
    
    fig, ax = plt.subplots()
    tools.plot_daynight_avg(dayT, nightT, ax, xlabel='Temperature [C]') 
    assert ax.get_ylabel() == 'Depth (m)'
    assert ax.get_xlabel() == xlabel
    assert ax.get_title() == sel_day

def test_temporal_drift():
    ds = fetchers.load_sample_dataset()
    fig, ax = plt.subplots(1, 2)
    tools.check_temporal_drift(ds,'DOXY', ax)
    assert ax.get_ylabel() == 'Depth (m)'
    assert ax.get_xlabel() == var
    assert ax.get_xlim == (np.nanpercentile(ds[var], 0.01), np.nanpercentile(ds[var], 99.99))
    tools.check_temporal_drift(ds,'CHLA')
        
def test_profile_check():
    ds = fetchers.load_sample_dataset()
    tools.check_monotony(ds.PROFILE_NUMBER)
    tools.plot_profIncrease(ds)

def test_check_monotony():
    ds = fetchers.load_sample_dataset()
    profile_number_monotony = tools.check_monotony(ds.PROFILE_NUMBER)
    temperature_monotony = tools.check_monotony(ds.TEMP)
    assert profile_number_monotony
    assert not temperature_monotony
    
