require 'nn'
require 'audio'
require 'mapper.Mapper'
require 'networks.UtilsMultiGPU'
local socket = require 'socket'
local cmd = torch.CmdLine()
cmd:option('-modelPath', 'deepspeech.t7', 'Path of model to load')
-- cmd:option('-audioPath', '', 'Path to the input audio to predict on')
cmd:option('-audioRoot', '', 'Path to the root folder of input audios')
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
if opt.nGPU > 0 then
    model = model:cuda()
end

local addr = '*'
local port = 5141

local server = assert(socket.bind(addr, port))
print('Bind: ' .. addr .. ':' .. tostring(port))

while true do
  local client = server:accept()
  client:settimeout(1)
  print('Accept: ' .. client:getpeername())
  while true do
    local content, status = client:receive()
    if status ~= 'closed' then
      if content then
        print('Recv: ' .. content)

        local filename = content  -- Maybe wrong, depends on implementation of luasocket

        -- predict
        local filepath = opt.audioRoot .. filename

        print('Load: ' .. filepath)
        local wave = audio.load(filepath)
        local spect = audio.spectrogram(wave, opt.windowSize * opt.sampleRate, 'hamming', opt.stride * opt.sampleRate):float() -- freq-by-frames tensor

        -- normalize the data
        local mean = spect:mean()
        local std = spect:std()
        spect:add(-mean)
        spect:div(std)

        spect = spect:view(1, 1, spect:size(1), spect:size(2))

        if opt.nGPU > 0 then
          spect = spect:cuda()
        end

        model:evaluate()
        local predictions = model:forward(spect)
        local text = mapper:decodeOutput(predictions[1])
        
	if text then
        	print('Result: ' .. text)

        	-- send result
        	client:send(text .. '\n')
	else
		client:send('*\n')
	end
      end
    else  -- disconnect
      break
    end
  end
  client:close()
end

mapper:closeDecoder()
server:close()  -- useless, please kill this process.
