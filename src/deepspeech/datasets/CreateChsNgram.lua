require 'torch'
require 'sys'
require 'xlua'
local paths = require 'paths'
local utils = require 'Utils'

local cmd = torch.CmdLine()
cmd:option('-textDir', 'thchs30/thchs30_dataset', 'Path to the dataset root')
cmd:option('-otherTextPath', '', 'Path to other text file for generating language model')
cmd:option('-outputTextPath', 'thchs30/ngram.txt', 'File containing the transcript to use')
cmd:option('-trainLanguageModel', false, 'whether train language model by the way')
cmd:option('-languageModelPath', 'thchs30/ngram.lm', ' File containing the language model to use')
cmd:option('-kenlmPath', '~/kenlm/build/bin/', 'Path to kenlm bin directory')
cmd:option('-n', 5, 'n gram order to use')
cmd:option('-prune', '0', 'whether to prune and how much')

local opt = cmd:parse(arg)
local splits = {'train', 'cv', 'test'}

local function printAndExec(command)
    print(command)
    sys.execute(command)
end

local sentences = {}
local function createTextForNgram(dirPath)
    local function compressText(textPath)
        for line in io.lines(textPath) do
            local text = utils.processChsText(line, ' ')
            sentences[text] = true
        end
    end
    local size = tonumber(sys.execute('ls ' .. dirPath .. ' | grep txt | wc -l'))
    local counter = 0
    for txt in paths.iterfiles(dirPath) do
        if string.find(txt, '.txt') then
            compressText(dirPath .. '/' .. txt)
            counter = counter + 1
            xlua.progress(counter, size)
        end
    end
    
end

for _, split in pairs(splits) do
    createTextForNgram(opt.textDir .. '/' .. split)
end

local file = io.open(opt.outputTextPath, "w")
for line, _ in pairs(sentences) do
    file:write(line)
    file:write("\n")
end
file:close()

print('already has dataset ngram text')

if not (opt.otherTextPath == '') then 
    print('preparing other text')
    printAndExec('mv ' .. opt.outputTextPath .. ' ' .. opt.outputTextPath .. '.bak')
    printAndExec('cat ' .. opt.outputTextPath .. '.bak ' .. opt.otherTextPath .. 
                    ' > ' .. opt.outputTextPath)
end

if opt.trainLanguageModel then
    printAndExec(opt.kenlmPath .. 'lmplz -o '.. opt.n .. 
                    ' -S 80% -T /tmp --prune ' .. opt.prune .. 
                    ' --discount_fallback < ' .. opt.textPath .. ' > ' .. 
                    opt.languageModelPath)
end