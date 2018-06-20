import matplotlib.pyplot as plt
import numpy


def loadGalaxyData(path):
    f = open(path, "rb")
    print("LOADING ", path, "...")
    data = numpy.load(f)
    print("SUCCESS!")
    f.close()
    assert isinstance(data, object)
    return data


def zeroNanValues(data):
    # remove nan by zeroing
    print("removing nan by zeroing...")
    for i in range(len(data)):
        for j in range(len(data[i])):
            if numpy.isnan(data[i][j]):
                data[i][j] = 0
    return data


def interpolateNanValuses(data):
    # remove nan by interpolation
    print("removing nan by interpolation...")
    for i in range(len(data)):
        INPUT_LEN = len(data[i])
        xp = [j for j in range(INPUT_LEN) if not numpy.isnan(data[i][j])]
        fp = [data[i][j] for j in range(INPUT_LEN) if not numpy.isnan(data[i][j])]
        dataNans = [j for j in range(INPUT_LEN) if numpy.isnan(data[i][j])]
        dataInterpolated = numpy.interp(dataNans, xp, fp)
        for j in range(len(dataNans)):
            data[i][dataNans[j]] = dataInterpolated[j]
        # verify we removed all nans
        for j in range(len(data[i])):
            if numpy.isnan(data[i][j]):
                print("data[%d][%d] is nan!!!" % (i, j))
    return data


def normalizeMedian(data):
    # normalize data using median
    print("normalizing data using median...")
    for i in range(len(data)):
        median_val = numpy.median(data[i])
        # print("median for %d is %d" % (i, median_val))
        data[i] /= median_val
    return data


def normalizeMax(data):
    # normalize data using max
    print("normalizing data using max...")
    for i in range(len(data)):
        max_val = numpy.max(data[i])
        # print("median for %d is %d" % (i, median_val))
        data[i] /= max_val
    return data


def decreaseRes(data, factor):
    newData = []
    for d in data:
        newD = []
        for v in range(int(len(data[0]) / factor)):
            value = numpy.median(d[v * factor:(v + 1) * factor])
            newD.append(value)
        newData.append(newD)
    return newData

def averageData(dataInput, averageDiv):
    layer_path_full = "LayerAverageW.npy"
    layerR_path_full = "LayerAverageWr.npy"

    waveLengthNum = len(dataInput[0])
    GALAXIES_NUMBER = len(dataInput)
    newWaveLengths = int(round( waveLengthNum/averageDiv,0))
    layerAverage = numpy.zeros(shape=(waveLengthNum,newWaveLengths))
    layerAverageRev = numpy.zeros(shape=(newWaveLengths, waveLengthNum))
    for j in range(newWaveLengths):
        for k in range(averageDiv):
            layerAverage[averageDiv*j+k,j] = 1/averageDiv
            layerAverageRev[j,averageDiv*j+k] = 1
    dataAveraged = numpy.dot(dataInput,layerAverage)
    dataDecode = numpy.dot(dataAveraged,layerAverageRev)
    dataErrorAverage = numpy.zeros(shape=(GALAXIES_NUMBER))
    for i in range(len(dataAveraged)):
        for j in range(len(dataAveraged[i])):
            dataErrorAverage[i] = dataErrorAverage[i] + (dataDecode[i][j] - data2[i][j])*(dataDecode[i][j] - data2[i][j])
    numpy.save(data_path_full, dataFull)
    numpy.save(data_path_full_error, dataErrorAverage)
    numpy.save(layer_path_full, layerAverage)
    numpy.save(layerR_path_full, layerAverageRev)
    f = open(path, "rb")
    print("LOADING ", path, "...")
    data = numpy.load(f)
    print("SUCCESS!")
    f.close()
    assert isinstance(data, object)
    return data


# D = loadGalaxyData("galaxyDataPart0.npy")
# D = normalizeMedian(zeroNanValues(D))
# ratio = 20
# D2 = decreaseRes(D, ratio)
# plt.plot(numpy.arange(3800, 9200, 0.5), D[0])
# plt.plot(numpy.arange(3800, 9200, 0.5 * ratio), D2[0])
# plt.pause(200)


def processData(nan = "INTERPOLATION", norm = "MEDIAN", files_num = 1):
    for i in range(files_num):
        print("### Processing galaxyDataPart%d.npy ###" % i)

        if nan == "ZEROING":  # remove nan by zeroing
            if norm == "MEDIAN":  # normalize by median
                D_zero_median = loadGalaxyData("galaxyDataPart%d.npy" % i)
                zeroNanValues(D_zero_median)
                normalizeMedian(D_zero_median)
                numpy.save("galaxyDataPart%d_zeroNan_medianNorm.npy" % i, D_zero_median)
                # verify:
                D_zero_median_load = numpy.load("galaxyDataPart%d_zeroNan_medianNorm.npy" % i)
                if (D_zero_median != D_zero_median_load).all():
                    print("D_zero_median != D_zero_median_load!")

            elif norm == "MAX":  # normalize by maximum
                D_zero_max = loadGalaxyData("galaxyDataPart%d.npy" % i)
                zeroNanValues(D_zero_max)
                normalizeMax(D_zero_max)
                numpy.save("galaxyDataPart%d_zeroNan_maxNorm.npy" % i, D_zero_max)
                # verify:
                D_zero_max_load = numpy.load("galaxyDataPart%d_zeroNan_maxNorm.npy" % i)
                if (D_zero_max != D_zero_max_load).all():
                    print("D_zero_max != D_zero_max_load!")

        elif nan == "INTERPOLATION":  # remove nan by interpolation
            if norm == "MEDIAN":  # normalize by median
                D_interp_median = loadGalaxyData("galaxyDataPart%d.npy" % i)
                interpolateNanValuses(D_interp_median)
                normalizeMedian(D_interp_median)
                numpy.save("galaxyDataPart%d_interpNan_medianNorm.npy" % i, D_interp_median)
                # verify:
                D_interp_median_load = numpy.load("galaxyDataPart%d_interpNan_medianNorm.npy" % i)
                if (D_interp_median != D_interp_median_load).all():
                    print("D_interp_median != D_interp_median_load!")

            elif norm == "MAX":  # normalize by maximum
                D_interp_max = loadGalaxyData("galaxyDataPart%d.npy" % i)
                interpolateNanValuses(D_interp_max)
                normalizeMax(D_interp_max)
                numpy.save("galaxyDataPart%d_interpNan_maxNorm.npy" % i, D_interp_max)
                # verify:
                D_interp_max_load = numpy.load("galaxyDataPart%d_interpNan_maxNorm.npy" % i)
                if (D_interp_max != D_interp_max_load).all():
                    print("D_interp_max != D_interp_max_load!")

    print("*************** DONE PROCESSING DATA ***************")
