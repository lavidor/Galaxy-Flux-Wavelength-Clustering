
import numpy
import tensorflow as tf
import math
import matplotlib.pyplot as plt

# set data processing variables
GALAXIES_NUM_FULL = 8998  # choose how many data files to load (1-9)
GALAXIES_NUM = 8998  # choose how many data files to load (1-9)
num_of_layers = 4
layers_dim = [540 , 128 , 32 , 2]

SAVE_PATHS = ["saves/interpolationLayer%d/model_interpolation.ckpt" % i for i in range(num_of_layers)]
CODE_PATHS = ["saves/interpolationLayer%d/model_coded.npy" % i for i in range(num_of_layers)]
DCOD_PATHS = ["saves/interpolationLayer%d/model_decoded.npy" % i for i in range(num_of_layers)]
COST_PATHS = ["saves/interpolationLayer%d/model_cost.npy" % i for i in range(num_of_layers)]
ENTROPY_PATHS = ["saves/interpolationLayer%d/model_entropy.npy" % i for i in range(num_of_layers)]
MAX_PATH = "saves/interpolationLayer0/model_maxNums.npy"
LAYER_CODE_PATHS = ["saves/interpolationLayer%d/model_codeLayers.npy" % i for i in range(num_of_layers)]
LAYER_DECD_PATHS = ["saves/interpolationLayer%d/model_decdLayers.npy" % i for i in range(num_of_layers)]
LAYER_CODEBIAS_PATHS = ["saves/interpolationLayer%d/model_codeBiasLayers.npy" % i for i in range(num_of_layers)]
LAYER_DECDBIAS_PATHS = ["saves/interpolationLayer%d/model_decdBiasLayers.npy" % i for i in range(num_of_layers)]

FINAL_PLOT_PATHS = ["cost_graph_layer%d.png" % i for i in range(num_of_layers)]
FINAL_WEIGHTS_CODE_PATHS = ["weights_code_layer%d.npy" % i for i in range(num_of_layers)]
FINAL_WEIGHTS_DECD_PATHS = ["weights_decode_layer%d.npy" % i for i in range(num_of_layers)]
FINAL_BIASES_CODE_PATHS = ["biases_decode_layer%d.npy" % i for i in range(num_of_layers)]
FINAL_BIASES_DECD_PATHS = ["biases_decode_layer%d.npy" % i for i in range(num_of_layers)]
FINAL_CODE_PATHS = ["model_coded_layer%d.npy" % i for i in range(num_of_layers)]
FINAL_DCOD_PATHS = ["model_decoded_layer%d.npy" % i for i in range(num_of_layers)]
FINAL_COST_PATHS = ["model_cost_layer%d.npy" % i for i in range(num_of_layers)]
FINAL_ENTROPY_PATHS = ["model_entropy_layer%d.npy" % i for i in range(num_of_layers)]
FINAL_MAX_PATH = "model_maxNums.npy"
data_max_full = "MaxNormalizedData.npy"
decoded_max_full = "MaxNormalizedDataDecoded.npy"
cost_full = "CostFull.npy"
entropy_full = "EntropyFull.npy"

maxNumbers = numpy.load(MAX_PATH)
numpy.save(FINAL_MAX_PATH, maxNumbers)
for i in range(4):
    codedData = numpy.load(CODE_PATHS[i])
    numpy.save(FINAL_CODE_PATHS[i], codedData)
    decodedData = numpy.load(DCOD_PATHS[i])
    numpy.save(FINAL_DCOD_PATHS[i], decodedData)
    costVector = numpy.load(COST_PATHS[i])
    numpy.save(FINAL_COST_PATHS[i], costVector)
    entropyVector = numpy.load(ENTROPY_PATHS[i])
    numpy.save(FINAL_ENTROPY_PATHS[i], entropyVector)
    wh = numpy.load(LAYER_CODE_PATHS[i])
    numpy.save(FINAL_WEIGHTS_CODE_PATHS[i], wh)
    wo = numpy.load(LAYER_DECD_PATHS[i])
    numpy.save(FINAL_WEIGHTS_DECD_PATHS[i], wo)
    bh = numpy.load(LAYER_CODEBIAS_PATHS[i])
    numpy.save(FINAL_BIASES_CODE_PATHS[i], bh)
    bo = numpy.load(LAYER_DECDBIAS_PATHS[i])
    numpy.save(FINAL_BIASES_DECD_PATHS[i], bo)

    xVector = numpy.zeros(len(costVector))
    for j in range(len(costVector)):
        xVector[j] = 100*j
    plt.ion()
    plt.ylabel("Average Squared Distance from input data to decoded data")
    plt.xlabel("Steps")
    plt.plot(xVector[1:], (costVector[1:]), ".g")
    plt.savefig(FINAL_PLOT_PATHS[i])
    plt.pause(0.05)
    plt.clf()
    plt.cla()
    plt.close()
codedData = numpy.load(FINAL_CODE_PATHS[3])
decodedData = numpy.load(FINAL_DCOD_PATHS[0])
originalData = numpy.load(data_path_full)
maxVector = numpy.load(FINAL_MAX_PATH)
costVector0 = numpy.load(FINAL_COST_PATHS[0])
costVector1 = numpy.load(FINAL_COST_PATHS[1])
costVector2 = numpy.load(FINAL_COST_PATHS[2])
costVector3 = numpy.load(FINAL_COST_PATHS[3])
xVector = numpy.zeros([5,max([len(costVector0),len(costVector1),len(costVector2),len(costVector3)])])
for j in range(len(xVector[0])):
    xVector[0][j] = 100*j
    if(j < len(costVector0)):
        xVector[1][j] = costVector0[j]
    if(j < len(costVector1)):
        xVector[2][j] = costVector1[j]
    if(j < len(costVector2)):
        xVector[3][j] = costVector2[j]
    if(j < len(costVector3)):
        xVector[4][j] = costVector3[j]
eVector0 = numpy.load(FINAL_ENTROPY_PATHS[0])
eVector1 = numpy.load(FINAL_ENTROPY_PATHS[1])
eVector2 = numpy.load(FINAL_ENTROPY_PATHS[2])
eVector3 = numpy.load(FINAL_ENTROPY_PATHS[3])
eVector = numpy.zeros([5,max([len(eVector0),len(eVector1),len(eVector2),len(eVector3)])])
for j in range(len(eVector[0])):
    eVector[0][j] = 100*j
    if(j < len(costVector0)):
        eVector[1][j] = eVector0[j]
    if(j < len(costVector1)):
        eVector[2][j] = eVector1[j]
    if(j < len(costVector2)):
        eVector[3][j] = eVector2[j]
    if(j < len(costVector3)):
        eVector[4][j] = eVector3[j]
for j in range(GALAXIES_NUM):
    originalData[j] = originalData[j]/maxVector[j]
    decodedData[j] = decodedData[j]/maxVector[j]
numpy.save(cost_full_full, xVector)
numpy.save(entropy_full_full, eVector)
numpy.save(data_max_full, originalData)
numpy.save(decoded_max_full, decodedData)
