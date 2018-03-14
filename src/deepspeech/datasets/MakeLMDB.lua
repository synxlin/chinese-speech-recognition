-- Expects data in the format of <root><train/test/cv><filename.wav/filename.txt>
-- Creates an LMDB of everything in these folders into train/test/cv set.

require 'lfs'
require 'audio'
require 'xlua'
require 'lmdb'
require 'torch'
require 'parallel'

local paths = require 'paths'
local tds = require 'tds'

local cmd = torch.CmdLine()
cmd:option('-rootPath', 'thchs30/thchs30_dataset', 'Path to the data')
cmd:option('-lmdbPath', 'thchs30/thchs30_lmdb', 'Path to save LMDBs to')
cmd:option('-windowSize', 0.02, 'Window size for audio data')
cmd:option('-stride', 0.01, 'Stride for audio data')
cmd:option('-sampleRate', 16000, 'Sample rate of audio data (Default 16khz)')
cmd:option('-audioExtension', 'wav', 'The extension of the audio files (wav/mp3/sph/etc)')
cmd:option('-processes', 8, 'Number of processes used to create LMDB')

local opt = cmd:parse(arg)
local rootPath = opt.rootPath
local lmdbRootPath = opt.lmdbPath
local extension = '.' .. opt.audioExtension
local splits = {train=true, test=true, cv=true}
parallel.nfork(opt.processes)

local function startWriter(path, name)
    local db = lmdb.env {
        Path = path,
        Name = name
    }
    db:open()
    local txn = db:txn()
    return db, txn
end

local function closeWriter(db, txn)
    txn:commit()
    db:close()
end

local function createLMDB(dataPath, lmdbPath)
    local vecs = tds.Vec()
    local size = tonumber(sys.execute("find " .. dataPath .. " -type f -name '*'" .. extension .. " | wc -l "))
    vecs:resize(size)
    
    local files = io.popen("find -L " .. dataPath .. " -type f -name '*" .. extension .. "'")
    local counter = 1
    print("Retrieving sizes for sorting...")
    local buffer = tds.Vec()
    buffer:resize(size)
    
    for file in files:lines() do
        buffer[counter] = file
        counter = counter + 1
    end
    
    local function getSize(opts)
        local audioFilePath = opts.file
        local transcriptFilePath = opts.file:gsub(opts.extension, ".txt")
        local opt = opts.opt
        local audioFile = audio.load(audioFilePath)
        local length = audio.spectrogram(audioFile, opt.windowSize * opt.sampleRate, 'hamming', opt.stride * opt.sampleRate):size(2)
        return { audioFilePath, transcriptFilePath, length }
    end
    
    for x = 1, opt.processes do
        local opts = { extension = extension, file = buffer[x], opt = opt }
        parallel.children[x]:send({ opts, getSize })
    end
    
    local processCounter = 1
    for x = 1, size do
        local result = parallel.children[processCounter]:receive()
        vecs[x] = tds.Vec(unpack(result))
        xlua.progress(x, size)
        if x % 1000 == 0 then collectgarbage() end
        -- send next index to retrieve
        if x + opt.processes <= size then
          local opts = { extension = extension, file = buffer[x + opt.processes], opt = opt }
          parallel.children[processCounter]:send({ opts, getSize })
        end
        if processCounter == opt.processes then
          processCounter = 1
        else
          processCounter = processCounter + 1
        end
    end
    print("Sorting...")
    -- sort the files by length
    local function comp(a, b) return a[3] < b[3] end
    
    vecs:sort(comp)
    local size = #vecs
    
    print("Creating LMDB dataset to: " .. lmdbPath)
    -- start writing
    local dbAudioPath, readerAudioPath = startWriter(lmdbPath .. '/audioPath', 'audioPath')
    local dbTrans, readerTrans = startWriter(lmdbPath .. '/trans', 'trans')
    
    for x = 1, size do
        local vec = vecs[x]
        local audioFilePath = vec[1]
        local transcriptFilePath = vec[2]
        local transcript
        for line in io.lines(transcriptFilePath) do
            transcript = line
        end
        
        readerAudioPath:put(x, 'datasets/' .. audioFilePath)
        readerTrans:put(x, transcript)
        
        -- commit buffer
        if x % 500 == 0 then
            readerAudioPath:commit(); readerAudioPath = dbAudioPath:txn()
            readerTrans:commit(); readerTrans = dbTrans:txn()
            collectgarbage()
        end
        xlua.progress(x, size)
    end
    
    closeWriter(dbAudioPath, readerAudioPath)
    closeWriter(dbTrans, readerTrans)
end

function parent()
    local function looper()
        require 'torch'
        require 'audio'
        while true do
            local object = parallel.parent:receive()
            local opts, code = unpack(object)
            local result = code(opts)
            parallel.parent:send(result)
            collectgarbage()
        end
    end
    
    parallel.children:exec(looper)
    
    for dir in paths.iterdirs(rootPath) do
        assert(splits[dir], dir .. " is not inside splits: train/test/cv")
        createLMDB(rootPath .. '/' .. dir, lmdbRootPath .. '/' .. dir)
    end
    
    parallel.close()
end

local ok, err = pcall(parent)
if not ok then
    print(err)
    parallel.close()
end