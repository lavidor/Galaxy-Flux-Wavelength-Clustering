import time
import matplotlib.pyplot as plt
import numpy
import tensorflow as tf

import data_utils

# set data processing variables
DATA_FILES_NUM = 9
GALAXIES_NUM = 8998

DATA_ORIGINAL_PATHS = ["galaxyDataPart%d.npy" % i for i in range(num_of_files)]
DATA_INTERPS_MEDIAN_PATHS = ["galaxyDataPart%d_interpNan_medianNorm.npy" % i for i in range(num_of_files)]
DATA_ZEROING_MEDIAN_PATHS = ["galaxyDataPart%d_zeroNan_medianNorm.npy" % i for i in range(num_of_files)]
DATA_INTERPS_MAX_PATHS = ["galaxyDataPart%d_interpNan_maxNorm.npy" % i for i in range(num_of_files)]
DATA_ZEROING_MAX_PATHS = ["galaxyDataPart%d_zeroNan_maxNorm.npy" % i for i in range(num_of_files)]
SAVE_PATH = "saves\\interpolation\\model_interpolation.ckpt"
PLOT_PATH = "saves\\interpolation\\plot_interpolation.png"
CODE_PATH = "saves\\interpolation\\model_coded.npy"
DCOD_PATH = "saves\\interpolation\\model_decoded.npy"
data_paths = ["galaxyDataPart%d_interpNan_medianNorm.npy" % i for i in range(DATA_FILES_NUM)]
data_path_full = "FullAveragedData.npy"
data_path_full_error = "DataErrorOfAverage.npy"
layer_path_full = "LayerAverageW.npy"
layerR_path_full = "LayerAverageWr.npy"

DATA_FILES_NUM = 9  # choose how many data files to load (1-9)
INTERPOLATION = True # If true, interpolate NAN values from surrounding values
ZEROING = False # If true, make any NAN value 0
MEDIAN = True # If true, normalize each vector by dividing by its median
MAX = False # If true, normalize each vector by dividing by its max
PROCESS_BOOL = False # False if the normalizing and NAN removal was done already

AVERAGE_BOOL = True # shorten wavelength vectors by averaging
averageDiv = 10 # save average of every averageDiv points in each vector

if ZEROING:  # remove nan by zeroing
    if MEDIAN:  # normalize by median
        if PROCESS_BOOL:
            data_utils.processData(nan="ZEROING", norm="MEDIAN", files_num=DATA_FILES_NUM)
        data_paths = DATA_ZEROING_MEDIAN_PATHS
    elif MAX:  # normalize by maximum
        if PROCESS_BOOL:
            data_utils.processData(nan="ZEROING", norm="MAX", files_num=DATA_FILES_NUM)
        sata_paths = DATA_ZEROING_MAX_PATHS
elif INTERPOLATION:  # remove nan by interpolation
    if MEDIAN:  # normalize by median
        if PROCESS_BOOL:
            data_utils.processData(nan="INTERPOLATION", norm="MEDIAN", files_num=DATA_FILES_NUM)
        data_paths = DATA_INTERPS_MEDIAN_PATHS
    elif MAX:  # normalize by maximum
        if PROCESS_BOOL:
            data_utils.processData(nan="INTERPOLATION", norm="MAX", files_num=DATA_FILES_NUM)
        data_paths = DATA_INTERPS_MAX_PATHS

if AVERAGE_BOOL:
    waveLengthNum = 10800
    data2 = numpy.zeros(shape=(0, waveLengthNum))
    for data_path in data_paths:
        data2 = numpy.concatenate((data2, numpy.load(data_path)))

    newWaveLengths = int(round( waveLengthNum/averageDiv,0))
    layerAverage = numpy.zeros(shape=(waveLengthNum,newWaveLengths))
    layerAverageRev = numpy.zeros(shape=(newWaveLengths, waveLengthNum))
    for j in range(newWaveLengths):
        for k in range(averageDiv):
            layerAverage[averageDiv*j+k,j] = 1/averageDiv
            layerAverageRev[j,averageDiv*j+k] = 1

    dataFull = numpy.dot(data2,layerAverage)
    dataDecode = numpy.dot(dataFull,layerAverageRev)
    dataErrorAverage = numpy.zeros(shape=(GALAXIES_NUM))
    for i in range(len(data2)):
        for j in range(len(data2[i])):
            dataErrorAverage[i] = dataErrorAverage[i] + (dataDecode[i][j] - data2[i][j])*(dataDecode[i][j] - data2[i][j])
    numpy.save(data_path_full, dataFull)
    numpy.save(data_path_full_error, dataErrorAverage)
    numpy.save(layer_path_full, layerAverage)
    numpy.save(layerR_path_full, layerAverageRev)

