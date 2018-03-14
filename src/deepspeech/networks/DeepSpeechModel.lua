require 'networks.UtilsMultiGPU'

local function RNNModule(inputDim, hiddenDim, opt)
    if opt.nGPU > 0 then
        if string.find(opt.rnnType, 'BN') then
            print('Using bidirection ' .. opt.rnnType ..' architecture')
            require 'networks.BatchBRNN'
            return cudnn.BatchBRNN(inputDim, hiddenDim, opt.rnnType)
        else
            local brnn = nn.Sequential()
            if opt.rnnType == 'gru' then
                print('Using bidirection GRU architecture')
                brnn:add(cudnn.BGRU(inputDim, hiddenDim, 1))
            else
                print('Using bidirection LSTM architecture')
                brnn:add(cudnn.BLSTM(inputDim, hiddenDim, 1))
            end
            brnn:add(nn.View(-1, 2, hiddenDim):setNumInputDims(2)) -- have to sum activations
            brnn:add(nn.Sum(3))
            return brnn
        end
    else
        require 'rnn'
        return nn.SeqBRNN(inputDim, hiddenDim)
    end
end

-- Creates the covnet+rnn structure.
local function deepSpeech(opt)
    local windowSize, sampleRate = 0.02, 16000
    local rnnInputsize = math.floor(windowSize * sampleRate / 2 + 1)
    local conv = nn.Sequential()
    -- (nInputPlane, nOutputPlane, kW, kH, [dW], [dH], [padW], [padH]) conv layers.
    conv:add(nn.SpatialConvolution(1, 32, 11, 41, 2, 2))
    conv:add(nn.SpatialBatchNormalization(32))
    conv:add(nn.Clamp(0, 20))
    rnnInputsize = math.floor((rnnInputsize - 41)/2 + 1)
    if opt.nbOfConvLayers == 2 then
        conv:add(nn.SpatialConvolution(32, 32, 11, 21, 1, 2))
        conv:add(nn.SpatialBatchNormalization(32))
        conv:add(nn.Clamp(0, 20))
        rnnInputsize = math.floor((rnnInputsize - 21)/2 + 1)
    end
    rnnInputsize = 32 * rnnInputsize -- based on the above convolutions and 16khz audio.
    local rnnHiddenSize = opt.hiddenSize -- size of rnn hidden layers
    local nbOfHiddenLayers = opt.nbOfHiddenLayers

    conv:add(nn.View(rnnInputsize, -1):setNumInputDims(3)) -- batch x features x seqLength
    conv:add(nn.Transpose({ 2, 3 }, { 1, 2 })) -- seqLength x batch x features

    local rnns = nn.Sequential()
    local rnnModule = RNNModule(rnnInputsize, rnnHiddenSize, opt)
    rnns:add(rnnModule:clone())
    rnnModule = RNNModule(rnnHiddenSize, rnnHiddenSize, opt)

    for i = 1, nbOfHiddenLayers - 1 do
        rnns:add(nn.Bottle(nn.BatchNormalization(rnnHiddenSize), 2))
        rnns:add(rnnModule:clone())
    end

    local fullyConnected = nn.Sequential()
    fullyConnected:add(nn.BatchNormalization(rnnHiddenSize))
    fullyConnected:add(nn.Linear(rnnHiddenSize, opt.dictionarySize))

    local model = nn.Sequential()
    model:add(conv)
    model:add(rnns)
    model:add(nn.Bottle(fullyConnected, 2))
    model:add(nn.Transpose({1, 2})) -- batch x seqLength x features
    model = makeDataParallel(model, opt.nGPU, opt.cudnnFastest)
    return model
end

-- Based on convolution kernel and strides.
local function calculateInputSizes1(sizes)
    sizes = torch.floor((sizes - 11) / 2 + 1) -- conv1
    return sizes
end

-- Based on convolution kernel and strides.
local function calculateInputSizes2(sizes)
    sizes = torch.floor((sizes - 11) / 2 + 1) -- conv1
    sizes = torch.floor((sizes - 11) / 1 + 1) -- conv2
    return sizes
end

return { deepSpeech, calculateInputSizes1, calculateInputSizes2 }