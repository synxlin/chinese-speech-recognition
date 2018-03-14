local Network = require 'Network'

-- Options can be overrided on command line run.
local cmd = torch.CmdLine()
cmd:option('-loadModel', false, 'Load previously saved model')
cmd:option('-loadPath', 'deepspeech.t7', 'Path to model to load')
cmd:option('-modelName', 'DeepSpeechModel', 'Name of class containing architecture')
cmd:option('-nGPU', 1, 'Number of GPUs, set -1 to use CPU')
cmd:option('-cudnnFastest', false, 'whether open cudnn fastest mode')
cmd:option('-trainingSetLMDBPath', './datasets/thchs30/thchs30_lmdb/train/', 'Path to LMDB training dataset')
cmd:option('-validationSetLMDBPath', './datasets/thchs30/thchs30_lmdb/cv/', 'Path to LMDB validation dataset')
cmd:option('-windowSize', 0.02, 'Window size for audio data')
cmd:option('-stride', 0.01, 'Stride for audio data')
cmd:option('-sampleRate', 16000, 'Sample rate of audio data (Default 16khz)')
cmd:option('-noiseAugmentation', false, 'whether training with noise augmentation')
cmd:option('-noiseInjection', false, 'whether training with noise injection')
cmd:option('-testNoiseAugmentation', false, 'whether testing with noise augmentation')
cmd:option('-testNoiseInjection', false, 'whether testing with noise injection')
cmd:option('-noiseRootPath', 'datasets/thchs30/noise', 'Path to the noise directory')
cmd:option('-logsTrainPath', './logs/TrainingLoss/', ' Path to save Training logs')
cmd:option('-logsValidationPath', './logs/ValidationScores/', ' Path to save Validation logs')
cmd:option('-epochSave', 0, 'save model every n epoch')
cmd:option('-modelTrainingPath', './models/', ' Path to save periodic training models')
cmd:option('-saveFileName', 'deepspeech.t7', 'Name of model to save as')
cmd:option('-dictionaryPath', './datasets/thchs30/dictionary', ' File containing the dictionary to use')
cmd:option('-languageModelPath', './datasets/thchs30/ngram.lm', ' File containing the language model to use')
cmd:option('-lmAlpha', 0, ' weight of language model probability | 0: not use language model')
cmd:option('-beamSize', 10, 'beam search size | 0: not use beam search')
cmd:option('-epochs', 70, 'Number of epochs for training')
cmd:option('-learningRate', 3e-4, ' Training learning rate')
cmd:option('-learningRateAnnealing', 1.1, 'Factor to anneal lr every epoch')
cmd:option('-maxNorm', 400, 'Max norm used to normalize gradients')
cmd:option('-momentum', 0.90, 'Momentum for SGD')
cmd:option('-batchSize', 5, 'Batch size in training')
cmd:option('-permuteBatch', false, 'Set to true if you want to permute batches AFTER the first epoch')
cmd:option('-validationBatchSize', 5, 'Batch size for validation')
cmd:option('-rnnType', 'lstm', 'RNN types: lstm | gru | lstmBN | gruBN | rnnReluBN')
cmd:option('-hiddenSize', 1024, 'RNN hidden sizes')
cmd:option('-nbOfHiddenLayers', 5, 'Number of rnn layers')
cmd:option('-nbOfConvLayers', 1, 'Number of conv layers: 1 | 2')

local opt = cmd:parse(arg)

--Parameters for the stochastic gradient descent (using the optim library).
local optimParams = {
    learningRate = opt.learningRate,
    learningRateAnnealing = opt.learningRateAnnealing,
    momentum = opt.momentum,
    dampening = 0,
    nesterov = true
}

--Create and train the network based on the parameters and training data.
Network:init(opt)

Network:trainNetwork(opt.epochs, optimParams)

--close Decoder
Network:closeDecoder()
--Creates the loss plot.
Network:createLossGraph()