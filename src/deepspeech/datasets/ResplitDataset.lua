require 'torch'
require 'sys'
local paths = require 'paths'
local utils = require 'Utils'

seed = 10
torch.manualSeed(seed)

local cmd = torch.CmdLine()
cmd:option('-rootPath', 'thchs30/thchs30_dataset', 'Path to the thchs30 root')
cmd:option('-audioExtension', 'wav', 'The extension of the audio files (wav/mp3/sph/etc)')
cmd:option('-copy', false, 'Copies the files over rather than moves for safety')

local opt = cmd:parse(arg)
local extension = '.' .. opt.audioExtension
local newPath = opt.rootPath .. '_new'
local splits = {'train', 'cv', 'test'}
local comm = opt.copy and 'cp ' or 'mv '

local function printAndExec(command)
    print(command)
    sys.execute(command)
end

-- get whole info of the dataset
local totalSize, dictionarySize = 0, 0
local totalFiles, utterId2Idx, utterIdx2Id, utterSizes = {}, {}, {}, {}
local dictionary, utterDictionarySize = {}, {}
for _, split in pairs(splits) do
    local dirPath = opt.rootPath .. '/' .. split
    printAndExec('ls ' .. dirPath .. ' |grep txt > tmp.txt')
    for line in io.lines('tmp.txt') do
        totalSize = totalSize + 1
        local fileName = paths.basename(line, '.txt')
        local id = sys.split(fileName, '_')
        local spkId, utterId = id[1], id[2]
        -- insert totalFiles
        local fileTable = { fileName = dirPath .. '/' .. fileName, spkId = spkId, utterId = utterId }
        table.insert(totalFiles, fileTable)
        -- record utter info
        if not utterSizes[utterId] then 
            utterSizes[utterId] = 0
            table.insert(utterIdx2Id, utterId)
            utterId2Idx[utterId] = #utterId2Idx
        end
        utterSizes[utterId] = utterSizes[utterId] + 1
        -- update dictionary
        local utterDic, utterDicSize = {}, 0
        local function addToDic(s)
            if not dictionary[s] then
                dictionary[s] ={}
                dictionarySize = dictionarySize + 1
            end
            dictionary[s][utterId] = true
            if not utterDic[s] then
                utterDic[s] = true
                utterDicSize = utterDicSize + 1
            end
        end
        
        for script in io.lines(dirPath .. '/' .. line) do
            utils.chsFilter(script, addToDic)
        end
        utterDictionarySize[utterId] = utterDicSize
    end
    printAndExec('rm tmp.txt')
end
for k, v in pairs(dictionary) do
    local tmp = {}
    for utID, _ in pairs(v) do
        table.insert(tmp, utID)
    end
    dictionary[k] = tmp
end
local utterSize = #utterIdx2Id
print('\n======\nThere are ' .. totalSize .. ' files')
print('There are ' .. utterSize .. ' utters')
print('There are ' .. dictionarySize .. ' items in dictionary')

-- expected trainSize, cvSize, testSize
local trainSize = math.ceil(totalSize * 0.8)
local testSize = totalSize - trainSize
local cvSize = math.ceil(testSize*0.25)

print('\n======\nExpected trainSize: ' .. trainSize)
print('Expected testSize: ' .. testSize - cvSize)
print('Expected cvSize: ' .. cvSize)

-- theoretically, training dataset should have complete dictionary
local function dicFromSmall(t, a, b)
    if #t[a] < #t[b] then
        return true
    elseif #t[a] == #t[b] then
        return a < b
    else
        return false
    end
end
local trainMustHaveUtters = {}
local trainUniqueUtterSize, trainUniqueUttersFileSize = 0, 0
for ch, uts in utils.spairs(dictionary, dicFromSmall) do
    local is_in = false
    local maxSize, maxId = 0
    for _, utID in pairs(uts) do
        if trainMustHaveUtters[utID] then
            is_in = true
            break
        end
        local utterDicSize = utterDictionarySize[utID]
        if utterDicSize > maxSize then
            maxSize = utterDicSize
            maxId = utID
        end
    end
    if not is_in then
        local size = utterSizes[maxId]
        trainMustHaveUtters[maxId] = size
        trainUniqueUttersFileSize = trainUniqueUttersFileSize + size
        trainUniqueUtterSize = trainUniqueUtterSize + 1
    end
end
print('\n======\nExpected Train set must have ' .. trainUniqueUtterSize .. ' unique utters')
print('Expected Train set will have ' .. trainUniqueUttersFileSize .. ' utters')
local coeff, tryTime = 1.0, 10
local is_trainSize_larger_trainUniqueUttersFileSize = trainSize > trainUniqueUttersFileSize
if not is_trainSize_larger_trainUniqueUttersFileSize then
    coeff = trainSize / trainUniqueUttersFileSize
end
while (not is_trainSize_larger_trainUniqueUttersFileSize) and tryTime > 0 do
    print('\n======\n' .. 11-tryTime .. ' time')
    tryTime = tryTime - 1
    coeff = coeff * 0.9
    print('Expected Train set will have ' .. trainUniqueUttersFileSize .. ' utters')
    print('However, Expected trainSize: ' .. trainSize)
    print('pruning coeff will be ' .. coeff)
    local trainCount = 0
    for k, v in pairs(trainMustHaveUtters) do
        local size = math.max(math.ceil(v * coeff), 1)
        trainCount = trainCount + size
    end
    is_trainSize_larger_trainUniqueUttersFileSize = trainSize > trainCount
end

if not is_trainSize_larger_trainUniqueUttersFileSize then
    print('error: cannot resplit dataset')
    return
else
    print('final pruning coeff is ' .. coeff)
    for k, v in pairs(trainMustHaveUtters) do
        local size = math.max(math.ceil(v * coeff), 1)
        trainMustHaveUtters[k] = size
    end
end

-- try to find proper test set which has unique utters and smaller than testSize
local testUniqueUtterSize = math.ceil(utterSize * 0.15)
testUniqueUtterSize = ((testUniqueUtterSize + trainUniqueUtterSize) < utterSize) and testUniqueUtterSize or math.ceil((utterSize - trainUniqueUtterSize)*0.8)
print('\n======\nExpected test set must have ' .. testUniqueUtterSize .. ' unique utters')
local permutedIndices = torch.randperm(utterSize)
local testUniqueUtters = {}

local is_testSize_larger_testUniqueUtterFileSize = false
tryTime = 10

while (not is_testSize_larger_testUniqueUtterFileSize) and tryTime > 0 do
    print('Try ' .. 11-tryTime .. ' time')
    tryTime = tryTime - 1
    local testUniqueUtterFileSize, testUniqueUttersCount = 0, 0
    testUniqueUtters = {}
    for i = 1, utterSize do
        if testUniqueUttersCount == testUniqueUtterSize then break end
        local utterId = utterIdx2Id[permutedIndices[i]]
        if not trainMustHaveUtters[utterId] then
            testUniqueUtters[utterId] = true
            testUniqueUtterFileSize = testUniqueUtterFileSize + utterSizes[utterId]
            testUniqueUttersCount = testUniqueUttersCount + 1
        end
    end
    if testSize > testUniqueUtterFileSize then
        is_testSize_larger_testUniqueUtterFileSize = true
        print('find a solution to test unique utters and test size')
    else
        permutedIndices = torch.randperm(utterSize)
    end
end

if not is_testSize_larger_testUniqueUtterFileSize then
    print('error: cannot resplit dataset')
    return
end

-- shuffle
local cvIndices = torch.randperm(testSize):sub(1, cvSize)
local totalIndices = torch.randperm(totalSize)

local trainResults, cvResults, testResults = {}, {}, {}
local alreadyAssignedIndices = {}
local trainCnt = 0

print('\n======\n Beigin resplit')
-- split the results
for i = 1, totalSize do
    local fileTable = totalFiles[i]
    local utterId = fileTable.utterId
    local trainMustHaveUttersRemain = trainMustHaveUtters[utterId]
    if trainMustHaveUttersRemain and trainMustHaveUttersRemain > 0 then
        trainCnt = trainCnt + 1
        table.insert(trainResults, fileTable)
        trainMustHaveUtters[utterId] = trainMustHaveUtters[utterId] - 1
        alreadyAssignedIndices[i] = true
    end
end
print('Alread Assigned ' .. trainCnt .. ' files to train set for its must-have utters')
for i = 1, totalSize do
    local idx = totalIndices[i]
    local fileTable = totalFiles[idx]
    local utterId = fileTable.utterId
    if not alreadyAssignedIndices[idx] then
        if not testUniqueUtters[utterId] and trainCnt < trainSize then
            trainCnt = trainCnt + 1
            table.insert(trainResults, fileTable)
        else
            table.insert(testResults, fileTable)
        end
        alreadyAssignedIndices[idx] = true
    end
end

for i = 1, cvSize do
    table.insert(cvResults, testResults[cvIndices[i]])
    testResults[cvIndices[i]] = nil
end

testSize = testSize - cvSize

-- save results
local results = {train=trainResults, cv=cvResults, test=testResults}
local sizes = {train=trainSize, cv=cvSize, test=testSize}
for _, split in pairs(splits) do
    local counter = 0
    local dirPath = newPath .. '/' .. split
    printAndExec('mkdir -p ' .. dirPath)
    local file = io.open(newPath .. '/' .. split .. '.scp', 'w')
    for _, v in pairs(results[split]) do
        file:write(v.fileName .. '\n')
        sys.execute(comm .. v.fileName .. '.* ' .. dirPath)
        counter = counter + 1
        xlua.progress(counter, sizes[split])
    end
    file:close()
    print('There are ' .. sizes[split] .. ' audio in train set')
end

print('\n========\ndone\n')
