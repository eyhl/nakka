import pyproj
import geopandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from netCDF4 import Dataset
from scipy.interpolate import RectBivariateSpline, interp1d
from datetime import datetime
'''
Function for creating flowline velocity figure for a given glacier.

Inputs:
    glacier:        string with name of glacier
    IV_file:        string with directory+name for velocity data. An .nc file
    CFL_file:       string with directory+name calving front line. A .shp file
    FL_file:        string with directory+name of flowline coordinates for the given glacier. A .txt file
    cm_file:        string with directory+name of colormap file. A .txt file
    outfile:        string with directory+name of output figure
    mean_IV_file:   string with directory+name of average velocity data. An .nc file

Creates a .png of flowline velocity plot.

Created by Øyvind Andreas Winton (oew) on 12 February 2020
Last edited by oew on 14 Apr 2020

'''

def IV_flowline_timeseries(glacier, IV_file, CFL_file, FL_file, cm_file, outfile, mean_IV_file):
    bed_file = '../MorlighemBed/BedMachineGreenland-2017-09-20.nc'  # Location of BecMachine data
    cfl = geopandas.read_file(CFL_file)                             #
    lon = np.array([])
    lat = np.array([])
    for i in range(cfl.geometry.shape[0]):  # Loop over all geometries and assume singleline geometry
        try:
            lon = np.append(lon, np.array(cfl.geometry[i].coords.xy[0]))
            lat = np.append(lat, np.array(cfl.geometry[i].coords.xy[1]))
            lon = np.append(lon, [np.nan])
            lat = np.append(lat, [np.nan])
        except NotImplementedError: # Loop over singleline geometries in the multiline geometry
            k = 0
            while ~np.isnan(k):
                try:
                    lon = np.append(lon, np.array(cfl.geometry[i][k].coords.xy[0]))
                    lat = np.append(lat, np.array(cfl.geometry[i][k].coords.xy[1]))
                    lon = np.append(lon, [np.nan])
                    lat = np.append(lat, [np.nan])
                    k+=1
                except ValueError:
                    k=np.nan
            print('NotImplementedError. Potentially that {} contains multi-part geometry in {}\'th part. A workaround implemented, but check for other possible errors.'.format(CFL_file, i))


    # CFL come in lat-lon, transform to polar stereographic (velocity data projection)
    inProj = pyproj.Proj(cfl.crs['init'])
    outProj = pyproj.Proj(proj='stere', datum="WGS84", lat_0=90, lat_ts=70, lon_0=-45, units="m")
    X_cfl_plot, Y_cfl_plot = pyproj.transform(inProj, outProj, lat, lon)
    X_cfl = X_cfl_plot[~np.isnan(X_cfl_plot)]
    Y_cfl = Y_cfl_plot[~np.isnan(Y_cfl_plot)]


    # Read flowline data
    fl = pd.read_csv(FL_file, sep='\s+', header=None, names=['lat', 'lon'])

    # Transform from lat-lon to polar stereographic, and reverse array to follow convention of calving front = 0km
    inProj = pyproj.Proj('epsg:4326')
    X_fl, Y_fl = pyproj.transform(inProj, outProj, fl.lat.values, fl.lon.values)
    X_fl, Y_fl = X_fl[::-1], Y_fl[::-1]

    # Make vector containing distance along flowline (starting at calving front)
    dX_fl = np.diff(X_fl)
    dY_fl = np.diff(Y_fl)
    d_fl = np.sqrt(dX_fl**2 + dY_fl**2) / 1000  # km. Distance between each point
    dist_fl = np.cumsum(d_fl) # Total distance along flowline
    dist_fl = np.insert(dist_fl, 0, 0)  # add 0 as first entry

    # Get latest IV data and interpolate to flow line
    IV = Dataset(IV_file, 'r')
    X_v = IV.variables['x'][:]
    Y_v = IV.variables['y'][:]
    v_variable = IV.variables['land_ice_surface_velocity_magnitude'][:]
    v_std_variable = IV.variables['land_ice_surface_velocity_magnitude_std'][:]
    mask = v_variable.mask[0, :, :]
    vel = v_variable.data[0, :, :]
    v_std = v_std_variable.data[0, :, :]
    v_int = RectBivariateSpline(X_v, Y_v[::-1], vel[::-1, :].T, kx=1, ky=1)         # More efficient alternative to interp2d. Note1: Requires strictly increasing coordinate vectors, thus [::-1] in both y-vector and v-matrix. Note2: Takes in z-values transposed relative interp1d, thus .T
    v_fl = v_int(X_fl, Y_fl, grid=False)
    v_fl[v_fl > 1e4] = np.nan

    # Get latest mean IV data and interpolate to flow line
    mean_IV = np.load(mean_IV_file)
    mean_v = mean_IV[0, :, :]
    mean_v[np.isnan(mean_v)] = 1e20
    mean_v_std = mean_IV[1, :, :]
    mean_v_std[np.isnan(mean_v_std)] = 1e20
    mean_v_int = RectBivariateSpline(X_v, Y_v[::-1], mean_v[::-1, :].T, kx=1, ky=1)
    mean_v_std_int = RectBivariateSpline(X_v, Y_v[::-1], mean_v_std[::-1, :].T, kx=1, ky=1)
    mean_v_fl = mean_v_int(X_fl, Y_fl, grid=False)
    mean_v_std_fl = mean_v_std_int(X_fl, Y_fl, grid=False)
    mean_v_fl[mean_v_fl > 1e4] = np.nan
    mean_v_std_fl[mean_v_fl > 1e4] = np.nan


    # Plot flowline velocities and put dates
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.set_xlabel('Distance from calving front [km]')
    ax1.set_ylabel('Ice velocity [m/day]')
    start_year = int(IV_file[-20:-16])
    start_month = int(IV_file[-16:-14])
    start_day = int(IV_file[-14:-12])
    start_date_str = datetime.strftime(datetime(start_year, start_month, start_day), "%Y %b %d")
    end_year = int(IV_file[-11:-7])
    end_month = int(IV_file[-7:-5])
    end_day = int(IV_file[-5:-3])
    end_date_str = datetime.strftime(datetime(end_year, end_month, end_day), "%Y %b %d")
    smb_start_year = int(mean_IV_file[-24:-20])
    smb_start_month = int(mean_IV_file[-20:-18])
    smb_start_str = datetime.strftime(datetime(smb_start_year, smb_start_month, 1), "%Y %b")
    smb_end_year = int(mean_IV_file[-12:-8])
    smb_end_month = int(mean_IV_file[-8:-6])
    smb_end_str = datetime.strftime(datetime(smb_end_year, smb_end_month, 1), "%Y %b")
    ax1.plot(dist_fl, v_fl, 'g', label=str(start_date_str) + ' - ' + str(end_date_str))
    ax1.plot(dist_fl, mean_v_fl, color='black', alpha=0.5, label='Mean velocity: ' + smb_start_str + ' - ' + smb_end_str)
    ax1.fill_between(dist_fl, mean_v_fl - mean_v_std_fl, mean_v_fl + mean_v_std_fl, alpha=0.1, color='black', label='Standard deviation of velocity: ' + smb_start_str + ' - ' + smb_end_str)
    ax1.set_title(glacier + ' ice flow velocity ' + str(start_date_str) + ' - ' + str(end_date_str))
    if glacier == '79F' or glacier == 'Storstrommen':
        ax1.legend(loc='upper right')
    else:
        ax1.legend(loc='upper left')
    try:
        # ax1.set_ylim([mean_v_fl[~np.isnan(mean_v_fl)].min()*0.2, mean_v_fl[~np.isnan(mean_v_fl)].max()*2])
        ax1.set_ylim([0, mean_v_fl[~np.isnan(mean_v_fl)].max()*2])
    except ValueError:
        # ax1.set_ylim([v_fl[~np.isnan(v_fl)].min()*0.2, v_fl[~np.isnan(v_fl)].max()*2])
        ax1.set_ylim([0, v_fl[~np.isnan(v_fl)].max()*2])
        print('No mean velocity on flowline.')
    # ax1.grid(b=True)


    ## Add inset map (IV+bedrock map)

    # Get extent of inset map and make square
    Y_min = np.minimum(Y_fl.min(), Y_cfl.min()) - 10000
    Y_max = np.maximum(Y_fl.max(), Y_cfl.max()) + 10000
    X_min = np.minimum(X_fl.min(), X_cfl.min()) - 10000
    X_max = np.maximum(X_fl.max(), X_cfl.max()) + 10000
    X_range = X_max - X_min
    Y_range = Y_max - Y_min
    if X_range < Y_range:
        correction = (Y_range - X_range)/2
        X_max += correction
        X_min -= correction
    elif X_range > Y_range:
        correction = (X_range - Y_range)/2
        Y_max += correction
        Y_min -= correction

    # Bedrock
    bedmachine = Dataset(bed_file, 'r')
    X_b = bedmachine.variables['x'][:].filled()
    Y_b = bedmachine.variables['y'][:].filled()
    bed = bedmachine.variables['bed'][:].filled()
    bed_mask = bedmachine.variables['mask'][:].filled()
    x_b_id = np.logical_and(X_b < X_max, X_b > X_min)
    y_b_id = np.logical_and(Y_b < Y_max, Y_b > Y_min)
    ocean = (bed_mask == 0)
    ocean = ocean[y_b_id][:, x_b_id]
    [x_ocean, y_ocean] = np.meshgrid(X_b[x_b_id], Y_b[y_b_id])

    # 2D velocity
    vel[mask] = np.nan  # Set masked values to nan
    np.warnings.filterwarnings('ignore')
    uncertainty_mask = v_std > 0.66 * vel # Set uncertain values to nan
    vel[uncertainty_mask] = np.nan
    cm_txt = pd.read_csv(cm_file, sep=',', header=None).values # Read colormap file
    cm_txt[0, 0] = 0.00001
    cm_txt[1, 0] = 0.001
    cmap = colors.ListedColormap(cm_txt[:, 1:4] / 255, name='IV_colormap') # create matplotlib colormap based on RGB values
    idx = np.arange(1, cm_txt.shape[0] + 1)
    vm_interp = interp1d(cm_txt[:, 0], idx, kind='nearest', bounds_error=False, fill_value=np.nan)
    vm = vm_interp(vel)
    vm[vm == 389] = np.nan  # Nans interpolate to 389 somewhy, here this is cancelled
    if glacier == '79F' or glacier == 'Storstrommen':
        ax2 = fig.add_axes([0.1, 0.6, 0.3, 0.3])
    else:
        ax2 = fig.add_axes([0.6, 0.6, 0.3, 0.3])
    ax2.set_aspect('equal', 'box')
    levels = [1, 2, 4, 6] + list(range(10, 400, 20))
    x_v_id = np.logical_and(X_v.filled() < X_max, X_v.filled() > X_min)
    y_v_id = np.logical_and(Y_v.filled() < Y_max, Y_v.filled() > Y_min)
    ax2.contourf(X_b[x_b_id], Y_b[y_b_id], bed[y_b_id][:, x_b_id], cmap=plt.cm.copper)
    cs = ax2.contourf(X_v[x_v_id], Y_v[y_v_id], vm[y_v_id][:,x_v_id], levels, cmap=cmap)
    ax2.scatter(x_ocean[ocean], y_ocean[ocean], color=plt.cm.tab10.colors[0], s=0.15)
    cbar = fig.colorbar(cs)
    cbar.set_label('Ice velocity [m/day]')
    cbar.set_ticks([1, 67, 235, 350])
    cbar.set_ticklabels(['0.01', '0.1', '1', '10'])
    cbar.set_clim([1, 389])

    # Plot calving front and flow line
    ax2.plot(X_cfl_plot, Y_cfl_plot, color='k', label='Calving front line')
    ax2.plot(X_fl, Y_fl, color='g', label='Flow line')

    # Set limits and remove ticks
    ax2.set_xlim([X_min, X_max])
    ax2.set_ylim([Y_min, Y_max])
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    ax2.set_yticks([])
    ax2.set_xticks([])

    ## Add scalebar #TODO Potentially there is some distortion going on here with the scale bar.
    scale_displace = Y_range * 0.035
    X_scale = [X_min + scale_displace, X_min + scale_displace + 10**4]
    Y_scale = [Y_min + scale_displace, Y_min + scale_displace]
    X_scale_end1 = [X_min + scale_displace, X_min + scale_displace]
    Y_scale_end1 = [Y_min + 0.8*scale_displace, Y_min + 1.2*scale_displace]
    X_scale_end2 = [X_min + scale_displace + 10**4, X_min + scale_displace + 10**4]
    Y_scale_end2 = [Y_min + 0.8*scale_displace, Y_min + 1.2*scale_displace]
    ax2.plot(X_scale, Y_scale, 'k')
    ax2.plot(X_scale_end1, Y_scale_end1, 'k')
    ax2.plot(X_scale_end2, Y_scale_end2, 'k')
    ax2.text(0, -0.02, '10km', ha='left', va='top', transform=ax2.transAxes)
    plt.tight_layout()
    plt.savefig(outfile, dpi=600)
    print('Done with IV {} for glacier  {}'.format(IV_file[-20:-3], glacier))
