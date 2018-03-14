require 'optim'
require 'nnx'
require 'gnuplot'
require 'lfs'
require 'xlua'
require 'nngraph'
require 'Loader'
require 'networks.UtilsMultiGPU'
require 'mapper.Mapper'
require 'ModelEvaluator'

local suffix = '_' .. os.date('%Y%m%d_%H%M%S')
local threads = require 'threads'
local Network = {}

--Training parameters
seed = 10
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(seed)

function Network:init(opt)
    self.fileName = opt.saveFileName
    self.nGPU = opt.nGPU
    self.gpu = self.nGPU > 0
    self.cudnnFastest = opt.cudnnFastest
    if not self.gpu then
        require 'rnn'
    else
        require 'cutorch'
        require 'cunn'
        require 'cudnn'
        require 'networks.BatchBRNN'
        cutorch.manualSeedAll(seed)
    end
    
    self.audioOpt = { sampleRate = opt.sampleRate,
                         windowSize = opt.windowSize,
                         stride = opt.stride,
                         noiseAugmentation = opt.noiseAugmentation,
                         noiseInjection = opt.noiseInjection,
                         testNoiseAugmentation = opt.testNoiseAugmentation,
                         testNoiseInjection = opt.testNoiseInjection,
                      }
    if opt.noiseInjection or opt.testNoiseInjection then
        self.audioOpt.noisePaths = {}
        for file in paths.iterfiles(opt.noiseRootPath) do
            table.insert(self.audioOpt.noisePaths, opt.noiseRootPath .. '/' .. file)
        end
    end
    self.trainingSetLMDBPath = opt.trainingSetLMDBPath
    self.validationSetLMDBPath = opt.validationSetLMDBPath
    self.logsTrainPath = opt.logsTrainPath or nil
    self.logsValidationPath = opt.logsValidationPath or nil
    self.modelTrainingPath = opt.modelTrainingPath or nil
    self.permuteBatch = opt.permuteBatch or false
    
    self:makeDirectories({ self.logsTrainPath, self.logsValidationPath, self.modelTrainingPath })
    
    self.mapper = Mapper(opt)
    self.tester = ModelEvaluator(self.gpu, self.validationSetLMDBPath, self.audioOpt, self.mapper, opt.validationBatchSize, self.logsValidationPath)
    self.loadModel = opt.loadModel
    self.epochSave = opt.epochSave or 0 -- Saves model every number of iterations.
    self.maxNorm = opt.maxNorm or 400 -- value chosen by Baidu for english speech.
    -- setting model saving/loading
    opt.dictionarySize = self.mapper.dictionarySize
    if self.loadModel then
        assert(opt.loadPath, "loadPath hasn't been given to load model.")
        self:loadNetwork(opt.loadPath, opt.modelName)
    else
        assert(opt.modelName, "Must have given a model to train.")
        self:prepSpeechModel(opt.modelName, opt)
    end
    -- setting online loading
    self.indexer = indexer(opt.trainingSetLMDBPath, opt.batchSize)
    self.pool = threads.Threads(1, function() require 'Loader' end)
    -- setting logger
    self.logger = optim.Logger(self.logsTrainPath .. 'train' .. suffix .. '.log')
    self.logger:setNames { 'loss', 'WER', 'CER' }
    self.logger:style { '-', '-', '-' }
end

function Network:prepSpeechModel(modelName, opt)
    local model = require('networks.' .. modelName)
    self.model = model[1](opt)
    local convLayerNum = 1
    if self.gpu then
        convLayerNum = #self.model:findModules('cudnn.SpatialConvolution')
    else
        convLayerNum = #self.model:findModules('nn.SpatialConvolution')
    end
    print('There are ' .. convLayerNum .. ' conv layer in the model')
    self.calSize = model[convLayerNum+1]
    collectgarbage("collect")
end

function Network:testNetwork(epoch)
    self.model:evaluate()
    local wer, cer = self.tester:runEvaluation(self.model, true, epoch or 1) -- details in log
    self.model:zeroGradParameters()
    self.model:training()
    return wer, cer
end

function Network:trainNetwork(epochs, optimizerParams)
    self.model:training()
    
    local lossHistory = {}
    local validationHistory = {}
    local criterion = nn.CTCCriterion(true)
    local x, gradParameters = self.model:getParameters()
    
    print("Number of parameters: ", gradParameters:size(1))
    
    -- inputs (preallocate)
    local inputs = torch.Tensor()
    local sizes = torch.Tensor()
    if self.gpu then
        criterion = criterion:cuda()
        inputs = inputs:cuda()
        sizes = sizes:cuda()
    end
    
    -- def loading buf and loader
    local loader = Loader(self.trainingSetLMDBPath, self.audioOpt, self.mapper)
    local specBuf, labelBuf, sizesBuf
    
    -- load first batch
    local inds = self.indexer:nextIndices()
    self.pool:addjob(function()
        return loader:nextBatch(inds)
    end,
        function(spect, label, sizes)
            specBuf = spect
            labelBuf = label
            sizesBuf = sizes
        end)
    
    -- define the feval
    local function feval(x_new)
        self.pool:synchronize() -- wait previous loading
        local inputsCPU, sizes, targets = specBuf, sizesBuf, labelBuf -- move buf to training data
        inds = self.indexer:nextIndices() -- load next batch whilst training
        self.pool:addjob(function()
            return loader:nextBatch(inds)
        end,
            function(spect, label, sizes)
                specBuf = spect
                labelBuf = label
                sizesBuf = sizes
            end)
        
        inputs:resize(inputsCPU:size()):copy(inputsCPU) -- transfer over to GPU
        sizes = self.calSize(sizes)
        local predictions = self.model:forward(inputs)
        local loss = criterion:forward(predictions, targets, sizes)
        if loss == math.huge or loss == -math.huge then loss = 0 print("Received an inf cost!") end
        self.model:zeroGradParameters()
        local gradOutput = criterion:backward(predictions, targets)
        self.model:backward(inputs, gradOutput)
        local norm = gradParameters:norm()
        if norm > self.maxNorm then
            gradParameters:mul(self.maxNorm / norm)
        end
        return loss, gradParameters
    end
    
    -- training
    local currentLoss
    local startTime = os.time()
    
    for i = 1, epochs do
        local averageLoss = 0
        
        for j = 1, self.indexer.nbOfBatches do
            currentLoss = 0
            local _, fs = optim.sgd(feval, x, optimizerParams)
            if self.gpu then cutorch.synchronize() end
            currentLoss = currentLoss + fs[1]
            xlua.progress(j, self.indexer.nbOfBatches)
            averageLoss = averageLoss + currentLoss
        end
        
        if self.permuteBatch then self.indexer:permuteBatchOrder() end
        
        averageLoss = averageLoss / self.indexer.nbOfBatches -- Calculate the average loss at this epoch.
        
        -- anneal learningRate
        optimizerParams.learningRate = optimizerParams.learningRate / (optimizerParams.learningRateAnnealing or 1)
        
        -- Update validation error rates
        local wer, cer = self:testNetwork(i)
        
        print(string.format("Training Epoch: %d Average Loss: %f Average Validation WER: %.2f Average Validation CER: %.2f",
            i, averageLoss, 100 * wer, 100 * cer))
        
        table.insert(lossHistory, averageLoss) -- Add the average loss value to the logger.
        table.insert(validationHistory, 100 * wer)
        self.logger:add { averageLoss, 100 * wer, 100 * cer }
        
        -- periodically save the model
        if self.epochSave > 0 and i % self.epochSave == 0 then
            print("Saving model..")
            self:saveNetwork(self.modelTrainingPath .. 'model_epoch_' .. i .. suffix .. '_' .. self.fileName)
        end
        
        collectgarbage("collect")
    end
    
    local endTime = os.time()
    local secondsTaken = endTime - startTime
    local minutesTaken = secondsTaken / 60
    print("Minutes taken to train: ", minutesTaken)
    
    print("Saving model..")
    self:saveNetwork(self.modelTrainingPath .. 'final_model_' .. suffix .. '_' .. self.fileName)
    
    return lossHistory, validationHistory, minutesTaken
end

function Network:closeDecoder()
    self.mapper:closeDecoder()
end

function Network:createLossGraph()
    self.logger:plot()
end

function Network:saveNetwork(saveName)
    self.model:clearState()
    saveDataParallel(saveName, self.model)
end

--Loads the model into Network.
function Network:loadNetwork(saveName, modelName)
    self.model = loadDataParallel(saveName, self.nGPU, self.cudnnFastest)
    local model = require('networks.' .. modelName)
    local convLayerNum = 1
    if self.gpu then
        convLayerNum = #self.model:findModules('cudnn.SpatialConvolution')
    else
        convLayerNum = #self.model:findModules('nn.SpatialConvolution')
    end
    print('There are ' .. convLayerNum .. ' conv layer in the model')
    self.calSize = model[convLayerNum+1]
    collectgarbage("collect")
end

function Network:makeDirectories(folderPaths)
    for index, folderPath in ipairs(folderPaths) do
        if (folderPath ~= nil) then os.execute("mkdir -p " .. folderPath) end
    end
end

return Network
