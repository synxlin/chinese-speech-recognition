require 'torch'
require 'xlua'
require 'Loader'
require 'mapper.Mapper'
require 'mapper.SequenceError'
local threads = require 'threads'

local ModelEvaluator = torch.class('ModelEvaluator')

local loader

function ModelEvaluator:__init(isGPU, datasetPath, opt, mapper, testBatchSize, logsPath)
    local audioOpt = { sampleRate = opt.sampleRate,
                         windowSize = opt.windowSize,
                         stride = opt.stride,
                         noiseAugmentation = opt.testNoiseAugmentation,
                         noiseInjection = opt.testNoiseInjection,
                         noisePaths = opt.noisePaths,
                       }
    loader = Loader(datasetPath, audioOpt, mapper)
    self.testBatchSize = testBatchSize
    self.nbOfTestIterations = math.ceil(loader.size / testBatchSize)
    self.indexer = indexer(datasetPath, testBatchSize)
    self.pool = threads.Threads(1, function() require 'Loader' end)
    self.mapper = mapper
    self.logsPath = logsPath
    self.suffix = '_' .. os.date('%Y%m%d_%H%M%S')
    self.sequenceError = SequenceError()
    self.input = torch.Tensor()
    self.isGPU = isGPU
    if isGPU then
        self.input = self.input:cuda()
    end
end

function ModelEvaluator:runEvaluation(model, verbose, epoch)
    local spect_buf, label_buf, sizes_buf, transcripts_buf
    
    -- get first batch
    local inds = self.indexer:nextIndices()
    self.pool:addjob(function()
        return loader:nextBatch(inds)
    end,
        function(spect, label, sizes, transcripts)
            spect_buf = spect
            label_buf = label
            sizes_buf = sizes
            transcripts_buf = transcripts
        end)
    
    if verbose then
        local f = assert(io.open(self.logsPath .. 'WER_Test' .. self.suffix .. '.log', 'a'), "Could not create validation test logs, does the folder "
                .. self.logsPath .. " exist?")
        f:write('======================== BEGIN WER TEST EPOCH: ' .. epoch .. ' =========================\n')
        f:close()
    end
    
    local evaluationPredictions = {} -- stores the predictions to order for log.
    local cumCER = 0
    local cumWER = 0
    local numberOfSamples = 0
    -- ======================= for every test iteration ==========================
    for i = 1, self.nbOfTestIterations do
        -- get buf and fetch next one
        self.pool:synchronize()
        local inputsCPU, targets, sizes_array, transcripts_array = spect_buf, label_buf, sizes_buf, transcripts_buf
        inds = self.indexer:nextIndices()
        self.pool:addjob(function()
            return loader:nextBatch(inds)
        end,
            function(spect, label, sizes, transcripts)
                spect_buf = spect
                label_buf = label
                sizes_buf = sizes
                transcripts_buf = transcripts
            end)
        
        self.input:resize(inputsCPU:size()):copy(inputsCPU)
        local predictions = model:forward(self.input)
        if self.isGPU then cutorch.synchronize() end
        
        local size = predictions:size(1)
        for j = 1, size do
            local prediction = predictions[j]
            local predictTranscript = self.mapper:decodeOutput(prediction)
            local targetTranscript = transcripts_array[j] --self.mapper:tokensToText(targets[j])
            
            local CER = self.sequenceError:calculateCER(targetTranscript, predictTranscript)
            local WER = self.sequenceError:calculateWER(targetTranscript, predictTranscript)
            
            cumCER = cumCER + CER
            cumWER = cumWER + WER
            
            table.insert(evaluationPredictions, { wer = WER * 100, cer = CER * 100, target = targetTranscript, prediction = predictTranscript })
        end
        numberOfSamples = numberOfSamples + size
        xlua.progress(i, self.nbOfTestIterations)
    end
    
    local function comp(a, b) return a.wer < b.wer end
    
    table.sort(evaluationPredictions, comp)
    
    if verbose then
        for index, eval in ipairs(evaluationPredictions) do
            local f = assert(io.open(self.logsPath .. 'Evaluation_Test' .. self.suffix .. '.log', 'a'))
            f:write(string.format("WER = %.2f | CER = %.2f | Text = \"%s\" | Predict = \"%s\"\n",
                eval.wer, eval.cer, eval.target, eval.prediction))
            f:close()
        end
    end
    local averageWER = cumWER / numberOfSamples
    local averageCER = cumCER / numberOfSamples
    
    local f = assert(io.open(self.logsPath .. 'Evaluation_Test' .. self.suffix .. '.log', 'a'))
    f:write(string.format("Average WER = %.2f | CER = %.2f", averageWER * 100, averageCER * 100))
    f:close()
    
    self.pool:synchronize() -- end the last loading
    return averageWER, averageCER
end