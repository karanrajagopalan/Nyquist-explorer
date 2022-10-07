import pandas as pd
import numpy as np
from netCDF4 import Dataset
import xarray as xr
import scipy.fft as fft
from scipy import signal
import base64
import io


def get_handle(file):
    nc_handle = Dataset(file, 'r')
    return nc_handle


def get_variable_data(nc_handle, key):
    # for key in nc_handle.variables.keys():
    return np.array(nc_handle[key][:]), nc_handle[key]


def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    # print(decoded)
    # print(contents)
    print(filename)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        elif 'nc' in filename:
            # temp = xr.open_dataset(io.BytesIO(decoded),engine='h5netcdf').load()
            ncdata = Dataset("in-mem-file",mode='r',memory=decoded)
            temp = xr.open_dataset(xr.backends.NetCDF4DataStore(ncdata))
            # print(temp)
            df = temp
            # print(df)
        return df
    except Exception as e:
        print(e)
    #     return None


def calculate_fft(array1, nfft, fs):
    fs = int(fs)
    nfft = int(nfft)
    fft_abs = pd.DataFrame(np.abs(fft.fft(array1, nfft)))
    fft_abs.columns = ['Amplitude']
    fft_abs['Frequency'] = np.linspace(0, fs, int(fs / (fs / nfft)))
    return fft_abs.head(int(nfft / 2))


def get_ftaps(f_type, f_order, fc_1, fc_2, fs):
    if fc_1 != 0 and fc_2 != 0:
        pass_zero = 'bandpass'
        cut_off = [x / (fs / 2) for x in [fc_1, fc_2]]
    elif fc_1 != 0:
        pass_zero = True
        cut_off = fc_1 / (fs / 2)
    elif fc_2 != 0:
        pass_zero = False
        cut_off = fc_2 / (fs / 2)

    return signal.firwin(f_order, cut_off, window=f_type, pass_zero=pass_zero)
