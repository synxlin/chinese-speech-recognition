require 'torch'
local paths = require 'paths'
local utils = require 'Utils'

local cmd = torch.CmdLine()
cmd:option('-textDir', 'thchs30/thchs30_dataset/train', 'Path to training directory where transcripts are')
cmd:option('-dictPath', 'thchs30/dictionary', 'Path to write dictionary file')
cmd:option('-countThreshold', 0, 'Threshold to prune the dictionary')

local opt = cmd:parse(arg)

local dictionary = {}

local function addToDic(s)
    if not dictionary[s] then
        dictionary[s] = 1
    else
        dictionary[s] = dictionary[s] + 1
    end
end

local function appendDictionary(dirPath)
    local sentences = {}
    local cnt = 0
    
    local function compressText(textPath)
        for line in io.lines(textPath) do
            local text = utils.processChsText(line)
            if not sentences[text] then
                sentences[text] = true
                cnt = cnt + 1
            end
        end
    end
    
    for textPath in paths.iterfiles(dirPath) do
        if string.find(textPath, '.txt') then
            compressText(dirPath .. '/' .. textPath)
        end
    end
    
    print('There are ' .. cnt .. ' unique sentences')
    
    local file = io.open(opt.dictPath .. '.text.txt', 'w')
    for line, _ in pairs(sentences) do
        file:write(line .. '\n')
        utils.chsFilter(line, addToDic)
    end
    file:close()
end

appendDictionary(opt.textDir)

local cnt = 0
local file = io.open(opt.dictPath, 'w')
file:write('$\n')
for ch, count in utils.svpairs(dictionary) do
    if count >= opt.countThreshold then
        file:write(ch .. '\n')
        cnt = cnt + 1
    end
end
file:write('*')
file:close()

print('There are ' .. cnt .. ' characters in the dictionary, excluding $ and *')
