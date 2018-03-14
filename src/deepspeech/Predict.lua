require 'nn'
require 'audio'
require 'mapper.Mapper'
require 'networks.UtilsMultiGPU'
local cmd = torch.CmdLine()
cmd:option('-modelPath', 'deepspeech.t7', 'Path of model to load')
cmd:option('-audioPath', '', 'Path to the input audio to predict on')
cmd:option('-dictionaryPath', './datasets/thchs30/dictionary', ' File containing the dictionary to use')
cmd:option('-languageModelPath', './datasets/thchs30/ngram.lm', ' File containing the language model to use')
cmd:option('-lmAlpha', 0, ' weight of language model probability | 0: not use language model')
cmd:option('-beamSize', 200, 'beam search size | 0: not use beam search')
cmd:option('-windowSize', 0.02, 'Window Size of audio')
cmd:option('-stride', 0.01, 'Stride of audio')
cmd:option('-sampleRate', 16000, 'Rate of audio (default 16khz)')
cmd:option('-nGPU', 1)
cmd:option('-cudnnFastest', false, 'whether open cudnn fastest mode')

local opt = cmd:parse(arg)

if opt.nGPU > 0 then
    require 'cunn'
    require 'cudnn'
    require 'networks.BatchBRNN'
end

local model =  loadDataParallel(opt.modelPath, opt.nGPU, opt.cudnnFastest)
local mapper = Mapper(opt)

local wave = audio.load(opt.audioPath)
local spect = audio.spectrogram(wave, opt.windowSize * opt.sampleRate, 'hamming', opt.stride * opt.sampleRate):float() -- freq-by-frames tensor

-- normalize the data
local mean = spect:mean()
local std = spect:std()
spect:add(-mean)
spect:div(std)

spect = spect:view(1, 1, spect:size(1), spect:size(2))

if opt.nGPU > 0 then
    spect = spect:cuda()
    model = model:cuda()
end

model:evaluate()
local predictions = model:forward(spect)
local text = mapper:decodeOutput(predictions[1])
mapper:closeDecoder()

print(text)