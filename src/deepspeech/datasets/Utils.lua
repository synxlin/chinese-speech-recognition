require 'torch'
require 'sys'
require 'audio'
local paths = require 'paths'

local utils = {}

-- chinese filter
function utils.chsFilter(line, func, except_symbols)
    local k = 1
    local except_bytes = {}
    if except_symbols then
        for _, c in pairs(except_symbols) do
            except_bytes[string.byte(c,1)] = true
        end
    end
    
    while true do  
        if k > #line then break end  
        local c = string.byte(line,k)  
        if not c then break end
        if except_bytes[c] then
            func(string.char(c))
            k = k + 1
        elseif c<192 then
            k = k + 1  
        elseif c<224 then  
            k = k + 2  
        elseif c<240 then  
            if c>=228 and c<=233 then  
                local c1 = string.byte(line,k+1)  
                local c2 = string.byte(line,k+2)  
                if c1 and c2 then  
                    local a1,a2,a3,a4 = 128,191,128,191  
                    if c == 228 then a1 = 184  
                    elseif c == 233 then a2,a4 = 190,c1 ~= 190 and 191 or 165  
                    end  
                    if c1>=a1 and c1<=a2 and c2>=a3 and c2<=a4 then
                        func(string.char(c, c1, c2))
                    end  
                end  
            end  
            k = k + 3  
        elseif c<248 then  
            k = k + 4  
        elseif c<252 then  
            k = k + 5  
        elseif c<254 then  
            k = k + 6  
        end  
    end
end

function utils.trim(s)
    return s:match'^()%s*$' and '' or s:match'^%s*(.*%S)'
end

-- strips down the chinse transcripts into pure text
function utils.processChsText(line, concatSymbol)
    local tokens = {}
    local function addToText(s)
        table.insert(tokens, s)
    end
    utils.chsFilter(line, addToText)
    local text = table.concat(tokens, concatSymbol)
    return utils.trim(text)
end

-- code from http://stackoverflow.com/questions/15706270/sort-a-table-in-lua
function utils.spairs(t, order)
    local keys = {}
    for k in pairs(t) do keys[#keys+1] = k end

    if order then
        table.sort(keys, function(a,b) return order(t, a, b) end)
    else
        table.sort(keys)
    end

    local i = 0
    return function()
        i = i + 1
        if keys[i] then
            return keys[i], t[keys[i]]
        end
    end
end

local function fs(t,a,b)
    if t[b] < t[a] then
        return true
    elseif t[b] == t[a] then
        return a < b
    else
        return false
    end
end

function utils.svpairs(t)
    return utils.spairs(t, fs)
end

function utils.getSpectrogram(audioPath, opt)
    local sampleRate = opt.sampleRate
    local windowSize = opt.windowSize
    local stride = opt.stride
    local noiseAug = opt.noiseAugmentation
    local noiseInj = opt.noiseInjection
    local noisePaths = opt.noisePaths
    local audioFile
    if noiseAug then
        local tmpAudioPath = 'tmp.wav'
        -- first tempo and gain (noise augmentation)
        local tempo = torch.uniform(0.8, 1.6)
        local gain = torch.uniform(-6, 8)
        local soxCommand = string.format('sox %s -r %d -c 1 -b 16 %s tempo %.3f gain %.3f', audioPath, sampleRate, tmpAudioPath, tempo, gain)
        sys.execute(soxCommand)
        audioFile = audio.load(tmpAudioPath)
        sys.execute('rm ' .. tmpAudioPath)
    else
        audioFile = audio.load(audioPath)
    end
    if noiseAug and noiseInj then
        -- then noise injection
        local noiseLevel = torch.uniform(0, 1)
        local noiseFile = audio.load(noisePaths[torch.random(1, #noisePaths)])
        local audioLen = audioFile:size(1)
        local noiseLen = noiseFile:size(1)
        local srcOffset = torch.random(1, noiseLen)
        local srcLeft = noiseLen - srcOffset + 1
        local dstOffset = 1
        local dstLeft = audioLen
        while dstLeft > 0 do
            local copySize = math.min(srcLeft, dstLeft)
            audioFile[{{dstOffset, dstOffset+copySize-1}}]:add(noiseLevel, noiseFile[{{srcOffset, srcOffset+copySize-1}}])
            if srcLeft > dstLeft then
                break
            else
                dstLeft = dstLeft - copySize
                dstOffset = dstOffset + copySize
                srcLeft = noiseLen
                srcOffset = 1
            end
        end
    end
    local spect = audio.spectrogram(audioFile, windowSize * sampleRate, 'hamming', stride * sampleRate)
    spect = spect:float()
    -- normalize the data
    local mean = spect:mean()
    local std = spect:std()
    spect:add(-mean)
    spect:div(std)
    return spect
end

return utils
