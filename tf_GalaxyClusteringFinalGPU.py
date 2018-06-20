
import numpy
import tensorflow as tf
import math
import matplotlib.pyplot as plt

# set data processing variables
GALAXIES_NUM_FULL = 8998  # number of galaxies available
GALAXIES_NUM = 8998  # choose how many galaxies to use
num_of_layers = 4 # Number of layers
layers_dim = [540 , 128 , 32 , 2] # Number of neurons in each layer

SAVE_PATHS = ["saves/interpolationLayer%d/model_interpolation.ckpt" % i for i in range(num_of_layers)] # Where to save the tensorflow session
PLOT_PATHS = ["saves/interpolationLayer%d/plot_interpolation.png" % i for i in range(num_of_layers)] # Where to save cost graphs
CODE_PATHS = ["saves/interpolationLayer%d/model_coded.npy" % i for i in range(num_of_layers)] # Where to save coded data
DCOD_PATHS = ["saves/interpolationLayer%d/model_decoded.npy" % i for i in range(num_of_layers)] # Where to save decoded data
COST_PATHS = ["saves/interpolationLayer%d/model_cost.npy" % i for i in range(num_of_layers)] # Where to save cost vector
ENTROPY_PATHS = ["saves/interpolationLayer%d/model_entropy.npy" % i for i in range(num_of_layers)] # Where to save entropy vector
MAX_PATH = "saves/interpolationLayer0/model_maxNums.npy" # Where to save max for each galaxy, for purpose of max normalization
LAYER_CODE_PATHS = ["saves/interpolationLayer%d/model_codeLayers.npy" % i for i in range(num_of_layers)] # Where to save encoding weights matrix
LAYER_DECD_PATHS = ["saves/interpolationLayer%d/model_decdLayers.npy" % i for i in range(num_of_layers)] # Where to save decoding weights matrix
LAYER_CODEBIAS_PATHS = ["saves/interpolationLayer%d/model_codeBiasLayers.npy" % i for i in range(num_of_layers)] # Where to save encoding biases vector
LAYER_DECDBIAS_PATHS = ["saves/interpolationLayer%d/model_decdBiasLayers.npy" % i for i in range(num_of_layers)] # Where to save decoding biases vector
layerAv_path_full = "LayerAverageW.npy"
layerAvR_path_full = "LayerAverageWr.npy"
data_path_full = "FullAveragedData.npy"
data_path_full_error = "DataErrorOfAverage.npy"
logitSmallReplacement = -4.4 # custom result of logit(x <= 0)
logitBigReplacement = 4.4 # custom result of logit(x >= 1)

l = 0 # Current layer to deal with
decode = True # Decode the next layer's decoded data if this is True, encode the previous layer's encoded data if False. Irrelevant if training is True
training = False # Train current layer if this is True
cont = True # Continue training session by loading previous session if this is True
ALPHA = 0.001 # For gradient descent
n_rounds = 101#1500000 # Number of steps for training
batch_size_max = 250 # Maximum number of galaxies to use in each step

if(l == 0): # If the layer is the first one, load the data and normalize it by maximum for each galaxy
    print("Loading data and normalizing by max")
    dataFull = numpy.load(data_path_full)
    data = numpy.zeros((GALAXIES_NUM , len(dataFull[0])))
    if(training==True and cont == False): # If this is the first training session, calculate the maxNums and save them
        maxNums = numpy.zeros(shape=(GALAXIES_NUM))
        for i in range(GALAXIES_NUM):
            maxNums[i] = max(dataFull[i])
        numpy.save(MAX_PATH, maxNums)
    else: # Use saved maxNums to normalize data
        maxNums = numpy.load(MAX_PATH)
    for i in range(GALAXIES_NUM): # data can contain numbers between 0 and 1
        data[i] = dataFull[i]/maxNums[i]
else:
    print("Loading encoded data from layer %d and calculating sigmoid of it" % (l-1))
    dataFull = numpy.load(CODE_PATHS[l-1]) # load the previous layer's encoded data
    data = numpy.zeros((GALAXIES_NUM , len(dataFull[0])))
    data = 1/(1+numpy.exp(-dataFull)) # data = sigmoid(dataFull) so that each number is between 0 and 1
input_data = data
output_data = data # Input and target decoded output is the same
n_samp, n_input = input_data.shape # n_samp is GALAXIES_NUM, n_input is previous layer's neuron count or 1080 if l is 0
n_hidden = layers_dim[l] # n_hidden is current layer's neuron count or 1080 if l is 0

print("Initializing session")
x = tf.placeholder("float", [None, n_input]) # placeholders for input
# Weights and biases from data to current layer encoding
Wh = tf.Variable(tf.random_uniform((n_input, n_hidden), -1.0 / math.sqrt(n_input), 1.0 / math.sqrt(n_input)))
bh = tf.Variable(tf.zeros([n_hidden]))
h = tf.matmul(x,Wh) + bh # the encoding function is: thie layer's encoding = (this layer's input) dot Wh + bh
# Weights and biases from current layer encoding to current layer decoding
Wo = tf.transpose(Wh) # tied weights
bo = tf.Variable(tf.zeros([n_input]))
y = tf.matmul(h,Wo) + bo # the decoding function is: thie layer's decoding = (this layer's encoding) dot Wo + bo)
# Added functionality to encode and decode (not used in code)
xInput = tf.Variable(tf.zeros([n_samp, n_input]))
hInput = tf.Variable(tf.zeros([n_samp, n_hidden]))
hOutput = tf.matmul(xInput,Wh) + bh
yOutput = tf.matmul(hInput,Wo) + bo
# Objective functions
y_ = tf.placeholder("float", [None,n_input]) # placeholders for decoded output
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
meansq = tf.reduce_mean(tf.square(y_-y))
train_step = tf.train.GradientDescentOptimizer(ALPHA).minimize(meansq) # minimize differences between input and decoded output
entropyVector = []
costVector = [] # Empty cost vector (cost is average squared difference between input and decoded output)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
batch_size = min(batch_size_max, n_samp) # Number of galaxies to use in each step
saver = tf.train.Saver()

if(training):
    if(cont): # Restore previous session for this layer
        print("Restoring previous session")
        saver.restore(sess, SAVE_PATHS[l])
        costVector = numpy.load(COST_PATHS[l]) # Continue previous cost vector
        entropyVector = numpy.load(ENTROPY_PATHS[l])
        costVector = costVector.tolist()
        entropyVector = entropyVector.tolist()
    print("Starting training, %d steps, batch size %d" % (n_rounds , batch_size))
    for i in range(n_rounds): 
        batch_xs = input_data[i % (n_samp - 1 - batch_size):i % (n_samp - 1 - batch_size) + batch_size][:] # Current batch is galaxies i to i+batch_size
        batch_ys = output_data[i % (n_samp - 1 - batch_size):i % (n_samp - 1 - batch_size) + batch_size][:]
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        if i % 100 == 0: # Calculate and print cost every 100 steps
            entropyVector.append(sess.run(cross_entropy, feed_dict={x: batch_xs, y_:batch_ys}))
            costVector.append(sess.run(meansq, feed_dict={x: batch_xs, y_:batch_ys}))
            print("Step %d, cost %f" % (i, costVector[len(costVector)-1]))

    print("Completed training, saving session")
    save_path = saver.save(sess, SAVE_PATHS[l]) # Training is done, save session
    dataDecoded = sess.run(y, feed_dict={x: input_data}) # Decoded output of layer
    if(l==0): # If this is layer 0, each decoded wavelength vector needs to be multiplied by the relevant max to match original input
        print("Denormalizing decoded output using saved maximums")
        for i in range(GALAXIES_NUM):
            dataDecoded[i] = maxNums[i]*dataDecoded[i]
    else: # If this is not layer 0, each decoded wavelength vector needs to be put through the inverse of sigmoid, logit, to match original data
        print("Calculating logit of decoded output")
        biggerIndices = (dataDecoded >= 1).nonzero() # logit works on numbers in range (0,1) only so if the decoded data went beyond range, it needs to be fixed
        smallerIndices = (dataDecoded <= 0).nonzero()
        dataDecoded[biggerIndices] = 0.5
        dataDecoded[smallerIndices] = 0.5
        dataDecoded = -numpy.log(numpy.power(dataDecoded, -1) - 1)
        dataDecoded[biggerIndices] = logitBigReplacement
        dataDecoded[smallerIndices] = logitSmallReplacement
    dataEncoded = sess.run(h, feed_dict={x: input_data}) # Encoded output of layer
    LayerEncode = numpy.zeros(shape=(n_input, n_hidden))
    LayerDecode = numpy.zeros(shape=(n_hidden, n_input))
    LayerEncodeBias = numpy.zeros(shape=(n_hidden))
    LayerDecodeBias = numpy.zeros(shape=(n_input))
    LayerEncode = sess.run(Wh) # Coding weights matrix of layer
    LayerEncodeBias = sess.run(bh) # Coding biases vector of layer
    LayerDecode = sess.run(Wo) # Decoding weights matrix of layer
    LayerDecodeBias = sess.run(bo) # Decoding biases vector of layer
    # Save results of training
    numpy.save(DCOD_PATHS[l], dataDecoded)
    numpy.save(CODE_PATHS[l], dataEncoded)
    numpy.save(LAYER_CODE_PATHS[l], LayerEncode)
    numpy.save(LAYER_DECD_PATHS[l], LayerDecode)
    numpy.save(LAYER_CODEBIAS_PATHS[l], LayerEncodeBias)
    numpy.save(LAYER_DECDBIAS_PATHS[l], LayerDecodeBias)
    numpy.save(COST_PATHS[l], costVector)
    numpy.save(ENTROPY_PATHS[l], entropyVector)
    print("Saving complete.")
    
else: # If training is False:
    print("Restoring previous session")
    saver.restore(sess, SAVE_PATHS[l])
    costVector = numpy.load(COST_PATHS[l])
    entropyVector = numpy.load(ENTROPY_PATHS[l])
    
    if(decode): # Deoode the decoded output of the next layer
        if(l < num_of_layers - 1): # There is no next layer if this is last layer
            print("Decoding the decoded output of layer %d using layer %d" % ((l+1), l))
            hInputLoad = numpy.load(DCOD_PATHS[l+1]) # hInputLoad is the decoded output of the next layer after logit
            WoCurrent = numpy.load(LAYER_DECD_PATHS[l]) # Decoding weights matrix of layer
            boCurrent = numpy.load(LAYER_DECDBIAS_PATHS[l]) # Decoding biases vector of layer
            dataDecoded = numpy.load(DCOD_PATHS[l])
            for i in range(GALAXIES_NUM): # Decode each vector in hInputLoad
                dataDecoded[i] = numpy.matmul(hInputLoad[i],WoCurrent) + boCurrent
            if (l==0): # If this is layer 0, each decoded wavelength vector needs to be multiplied by the relevant max to match original input
                print("Denormalizing decoded output using saved maximums")
                for i in range(GALAXIES_NUM):
                    dataDecoded[i] = maxNums[i]*dataDecoded[i]
            else: # If this is not layer 0, each decoded wavelength vector needs to be put through the inverse of sigmoid, logit, to match original data
                print("Calculating logit of decoded output")
                biggerIndices = (dataDecoded >= 1).nonzero() # logit works on numbers in range (0,1) only so if the decoded data went beyond range, it needs to be fixed
                smallerIndices = (dataDecoded <= 0).nonzero()
                dataDecoded[biggerIndices] = 0.5
                dataDecoded[smallerIndices] = 0.5
                dataDecoded = -numpy.log(numpy.power(dataDecoded, -1) - 1)
                dataDecoded[biggerIndices] = logitBigReplacement
                dataDecoded[smallerIndices] = logitSmallReplacement
            numpy.save(DCOD_PATHS[l], dataDecoded) # Save decoded data
            print("Saving complete.")
        else:
            print("Cannot decode layer %d" % (l+1))
    else: # Encode the encoded output of the previous layer
        if(l == 0): # Encode the original data using layer 0
            print("Encoding normalized data using layer %d" % l)
            xInputLoad = numpy.load(data_path_full) # xInputLoad is the original data
            for i in range(GALAXIES_NUM): # Normalize the original data by max
                xInputLoad[i] = xInputLoad[i]/max(xInputLoad[i])
        else: # Encode the encoded output of the previous layer
            print("Encoding the sigmoid of the encoded output of layer %d using layer %d" % ((l-1), l))
            xInputLoad = numpy.load(CODE_PATHS[l-1]) # xInputLoad is the encoded output of the previous layer
            xInputLoad = 1/(1+numpy.exp(-xInputLoad)) # xInputLoad = sigmoid(xInputLoad) so that each number is between 0 and 1
        WhCurrent = numpy.load(LAYER_CODE_PATHS[l]) # Encoding weights matrix of layer
        bhCurrent = numpy.load(LAYER_CODEBIAS_PATHS[l]) # Encoding biases vector of layer
        dataEncoded = numpy.load(CODE_PATHS[l])
        for i in range(GALAXIES_NUM): # Decode each vector in xInputLoad
            dataEncoded[i] = numpy.matmul(xInputLoad[i],WhCurrent) + bhCurrent
        numpy.save(CODE_PATHS[l], dataEncoded) # Save encoded data
        if(l == num_of_layers-1):  # If this is the last layer, decode the encoding
            print("Decoding the encoded output of layer %d using layer %d" % (l, l))
            WoCurrent = numpy.load(LAYER_DECD_PATHS[l])
            boCurrent = numpy.load(LAYER_DECDBIAS_PATHS[l])
            dataDecoded = numpy.load(DCOD_PATHS[l])
            for i in range(GALAXIES_NUM):
                dataDecoded[i] = numpy.matmul(dataEncoded[i],WoCurrent) + boCurrent
            print("Calculating logit of decoded output")
            biggerIndices = (dataDecoded >= 1).nonzero() # logit works on numbers in range (0,1) only so if the decoded data went beyond range, it needs to be fixed
            smallerIndices = (dataDecoded <= 0).nonzero()
            dataDecoded[biggerIndices] = 0.5
            dataDecoded[smallerIndices] = 0.5
            dataDecoded = -numpy.log(numpy.power(dataDecoded, -1) - 1) # logit the decoded data so it matches the previous layer's encoded output
            dataDecoded[biggerIndices] = logitBigReplacement
            dataDecoded[smallerIndices] = logitSmallReplacement
            numpy.save(DCOD_PATHS[l], dataDecoded) # Save decoded data
        print("Saving complete.")

