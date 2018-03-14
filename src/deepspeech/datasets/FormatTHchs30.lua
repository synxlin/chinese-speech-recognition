require 'torch'
require 'sys'
local threads = require 'threads'

local cmd = torch.CmdLine()
cmd:option('-rootPath', 'thchs30/thchs30', 'Path to the thchs30 root')
cmd:option('-audioExtension', 'wav', 'The extension of the audio files (wav/mp3/sph/etc)')
cmd:option('-move', false, 'Moves the files over rather than copies, used to save space')
cmd:option('-threads', 8, 'Number of threads to use')
cmd:option('-noise', false, 'whether to add noise data to dataset')

local opt = cmd:parse(arg)
local extension = '.' .. opt.audioExtension
local newPath = opt.noise and opt.rootPath .. '_noise_dataset' or opt.rootPath .. '_dataset'

local thchs30Splits = {'train', 'cv', 'test'}
local thchs30NoiseSplits = {'cafe', 'car', 'white', 'mix'}
local thchs30AudioPath = {
        train = opt.rootPath .. '/wav/train',
        cv = opt.rootPath .. '/wav/train',
        test = opt.rootPath .. '/wav/test'
    }
local thchs30NoiseDirPath = {
        train = opt.rootPath .. '/wav/train-noise',
        cv = opt.rootPath .. '/wav/cv-noise',
        test = opt.rootPath .. '/wav/test-noise'
    }
local thchs30NoiseAudioPath = {
        train = opt.rootPath .. '/wav/train-noise/0db',
        cv = opt.rootPath .. '/wav/cv-noise/0db',
        test = opt.rootPath .. '/wav/test-noise/0db'
    }
local thchs30TransPath = opt.rootPath .. '/doc/trans'

local threads = threads.Threads(opt.threads, function(idx)
                                                        require 'torch'
                                                        require 'sys'
                                                        paths = require 'paths'
                                                        utils = require 'Utils'
                                                    end)

local function createDataset(thchs30AudioPath, split, noiseSplit)
    local audioDirPath = thchs30AudioPath
    audioDirPath = noiseSplit and audioDirPath .. '/' .. noiseSplit or audioDirPath
    if not paths.dirp(audioDirPath) then
        print('warning: ' .. audioDirPath .. ' not exist - skip')
        return
    end
    
    local newDirPath = newPath .. '/' .. split
    sys.execute("mkdir -p " .. newDirPath)
    
    local transcriptPath = thchs30TransPath .. '/' .. split .. '.word.txt'
    local size = tonumber(sys.execute('cat ' .. transcriptPath .. ' | wc -l '))

    local function formatData(line)
        local text = utils.processChsText(line)
        local id = string.gmatch(line, '[^%s]+'){1} -- first part of transcript, used for audio file path and ID
        local audioFolders = sys.split(id, '_') -- [1] is the directory name
        local secondLevel = noiseSplit and noiseSplit or audioFolders[1]
        local textPath = newDirPath .. '/' .. id .. '_' .. secondLevel .. '.txt'
        local audioPath = thchs30AudioPath .. '/' .. secondLevel .. '/' .. id
        local newAudioPath = newDirPath .. '/' .. id .. '_' .. secondLevel .. extension
        
        local file = io.open(textPath, 'w')
        file:write(text)
        file:close()
        -- move/copy audio to correct place
        local command = opt.move and 'mv ' or 'cp '
        if paths.filep(audioPath .. extension) then
            sys.execute(command .. audioPath .. extension .. ' ' .. newAudioPath)
        else
            sys.execute(command .. audioPath .. string.upper(extension) .. ' ' .. newAudioPath)
        end
    end

    local counter = 0
    for line in io.lines(transcriptPath) do
        threads:addjob(function()
                              formatData(line)
                          end,
                          function()
                              counter = counter + 1
                              xlua.progress(counter, size)
                          end)
    end
end

sys.execute('mkdir -p ' .. newPath)

if opt.noise then
    local paths = require 'paths'
    -- check whether data is complete (with train/cv/test-noise, since origin tgz file only contains cv/test-noise (cv-noise is named after train-noise))
    if not paths.dirp(thchs30NoiseDirPath['cv']) then
        sys.execute('th GenTHchs30Noise.lua -rootPath ' .. opt.rootPath)
    end
end

for _, split in pairs(thchs30Splits) do 
    createDataset(thchs30AudioPath[split], split)
    if opt.noise then
        for _, noiseSplit in pairs(thchs30NoiseSplits) do
            createDataset(thchs30NoiseAudioPath[split], split, noiseSplit)
        end
    end
end
