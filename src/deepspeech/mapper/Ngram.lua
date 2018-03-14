require 'torch'
require 'sys'
require 'parallel'
local sk = require 'socket'
local paths = require 'paths'

-- construct an object to deal with n gram language model
local ngramLM = torch.class('Ngram')

function ngramLM:__init(opt, token2alphabet, dictionarySize, blankToken)
    self.lmAlpha = opt.lmAlpha
    self.GPU = opt.nGPU > 0
    self.token2alphabet = token2alphabet
    self.dictionarySize = dictionarySize
    self.blankToken = blankToken
    self.host = "127.0.0.1"
    self.port = 50963
    
    local lmPath = opt.languageModelPath
    assert(paths.filep(lmPath), lmPath .. ' not found')
    -- start python server
    local function worker()
        local lmPath = parallel.parent:receive()
        require 'os'
        print('python mapper/NgramServer.py --languageModelPath ' .. lmPath)
        os.execute('python mapper/NgramServer.py --languageModelPath ' .. lmPath)
    end
    
    local function parent()
        parallel.nfork(1)
        parallel.children:exec(worker)
        parallel.children:send(lmPath)
    end
    
    local ok, err = pcall(parent)
    if not ok then print(err); parallel.close(); end
    
    -- begin communication
    while not paths.filep('mapper/ngram.info.log') do
    end
    local client = assert(sk.connect(self.host, self.port))
    client:send("pid")
    self.pid = client:receive('*a')
    client:close()
    print("Successfully communicate with ngram.py - pid: " .. self.pid)
end

function ngramLM:closeServer()
    print('closing NgramServer')
    sys.execute('kill ' .. self.pid)
    sys.execute('rm -f mapper/ngram.info.log')
    parallel.close()
end

function ngramLM:getScore(sentence)
    --[[
        Input: string sentence
    --]]
    local host, port = self.host, self.port
    local client = assert(socket.connect(host, port))
    client:send(sentence)
    local score = client:receive('*a')
    client:close()
    score = tonumber(score)
    return score
end

function ngramLM:getMaxSentence(tokens, ctcScores)
    --[[
        Input: table of ctc tokens, size: beamSize
               torch Tensor of ctcScores, size: beamSize
    --]]
    local sentences = {}
    local scores = {}
    local blankToken = self.blankToken
    local token2alphabet = self.token2alphabet
    local host, port = self.host, self.port
    local function getScore(localTokens)
        local sentenceSpace = ""
        local sentence = ""
        local preToken = blankToken
        for _, token in pairs(localTokens) do
            if token ~= blankToken and token ~= preToken then
                sentenceSpace = sentenceSpace .. token2alphabet[token] .. ' '
                sentence = sentence .. token2alphabet[token]
            end
            preToken = token
        end
        local client = assert(socket.connect(host, port))
        client:send(sentenceSpace)
        local score = client:receive('*a')
        client:close()
        return sentence, score
    end

    for i, localTokens in pairs(tokens) do
        if #localTokens == 0 then
            scores[i] = -math.huge
            sentences[i] = ""
        else
            local sentence, score = getScore(localTokens)
            scores[i] = tonumber(score)
            sentences[i] = sentence
        end
    end
    
    local maxScore = -math.huge
    local maxIdx = 0
    for i, score in pairs(scores) do
        local totalScore = self.lmAlpha * score + ctcScores[i]
        if totalScore > maxScore then
            maxIdx = i
            maxScore = totalScore
        end
    end
    local sentence = sentences[maxIdx]
    local score = maxScore
    sentences, token2alphabet, scores = nil, nil, nil
    collectgarbage("collect")
    return sentence, score
end

function ngramLM:getMaxPrediction(hisSentenceSpace, hisSentence, ctcScores)
    --[[
        Input: string, history sentence with space
               string, history sentence
               torch Tensor of ctcScores, size: dictionarySize
    --]]
    local lmAlpha = self.lmAlpha
    local sentences, scores, sentencesSpace = {}, {}, {}
    local token2alphabet = self.token2alphabet
    local host, port = self.host, self.port
    local function getScore(curToken)
        local sentenceSpace = hisSentenceSpace ..' ' .. token2alphabet[curToken]
        local sentence = hisSentence .. token2alphabet[curToken]
        local client = assert(socket.connect(host, port))
        client:send(sentenceSpace)
        local score = client:receive('*a')
        client:close()
        return sentenceSpace, sentence, score
    end
    
    for i = 1, self.dictionarySize do
        if i-1 == self.blankToken then
            scores[i] = -math.huge
            sentences[i] = hisSentence
            sentencesSpace[i] = hisSentenceSpace
        else
            local sentenceSpace, sentence, score = getScore(i-1)
            scores[i] = score
            sentences[i] = sentence
            sentencesSpace[i] = sentenceSpace
        end
    end
    
    local maxScore = -math.huge
    local maxIdx = 0
    for i, score in pairs(scores) do
        if not (i-1 == self.blankToken) then
            local totalScore = lmAlpha * score + ctcScores[i]
            if totalScore > maxScore then
                maxIdx = i
                maxScore = totalScore
            end
         end
    end
    local sentenceSpace = sentencesSpace[maxIdx]
    local score = maxScore
    local sentence = sentences[maxIdx]
    sentencesSpace, token2alphabet, scores, sentences = nil, nil, nil, nil
    collectgarbage("collect")
    return sentenceSpace, sentence, score
end

function ngramLM:getBeamMaxPrediction(hisSentenceSpace, hisSentence, ctcScores, beamSize)
    --[[
        Input: string, history sentence with space
               string, history sentence
               torch Tensor of ctcScores, size: dictionarySize
               number, beamSize
    --]]
    local lmAlpha = self.lmAlpha
    local sentences, scores, sentencesSpace = {}, {}, {}
    local token2alphabet = self.token2alphabet
    local host, port = self.host, self.port
    local function getScore(curToken)
        local sentenceSpace = hisSentenceSpace ..' ' .. token2alphabet[curToken]
        local sentence = hisSentence .. token2alphabet[curToken]
        local client = assert(socket.connect(host, port))
        client:send(sentenceSpace)
        local score = client:receive('*a')
        client:close()
        return sentenceSpace, sentence, score
    end
    
    local _, maxIndices = torch.topk(ctcScores, beamSize, 1, true)
    maxIndices = maxIndices:float():squeeze()
    
    for i = 1, beamSize do
        local idx = maxIndices[i]
        local curToken = idx - 1
        local ctcScore = ctcScores[idx]
        if not (curToken == self.blankToken) then
            local sentenceSpace, sentence, score = getScore(curToken)
            scores[i] = ctcScore + lmAlpha * score
            sentences[i] = sentence
            sentencesSpace[i] = sentenceSpace
        end
    end
    
    local maxScore = -math.huge
    local maxIdx = 0
    for i, score in pairs(scores) do
        if score > maxScore then
            maxIdx = i
            maxScore = score
        end
    end
    local sentenceSpace = sentencesSpace[maxIdx]
    local score = maxScore
    local sentence = sentences[maxIdx]
    sentencesSpace, token2alphabet, scores, sentences, maxIndices = nil, nil, nil, nil, nil
    collectgarbage("collect")
    return sentenceSpace, sentence, score
end

function ngramLM:getBeamPredictions(hisSentencesSpace, hisSentences, ctcScores, hisCtcScores, beamSize)
    --[[
        Input: table of string, history sentence with space
               table of string, history sentence
               torch Tensor of ctcScores, size: dictionarySize
               table of number, history sentences CTC scores
               number, beamSize
    --]]
    local lmAlpha = self.lmAlpha
    local token2alphabet = self.token2alphabet
    local host, port = self.host, self.port
    local function getScore(curToken, hisSentenceSpace, hisSentence)
        local sentenceSpace = hisSentenceSpace ..' ' .. token2alphabet[curToken]
        local sentence = hisSentence .. token2alphabet[curToken]
        local client = assert(socket.connect(host, port))
        client:send(sentenceSpace)
        local score = client:receive('*a')
        client:close()
        return sentenceSpace, sentence, score
    end
    
    local sentences, scores, newCtcScores, sentencesSpace = {}, {}, {}, {}
    local _, maxIndices = torch.topk(ctcScores, beamSize+1, 1, true) --(beamSize+1 in case of blankToken)
    maxIndices = maxIndices:float():squeeze()
    
    for i = 1, beamSize do -- iterate hisSentences & hisSentencesSpace
        local hisSentenceSpace = hisSentencesSpace[i]
        local hisSentence = hisSentences[i]
        local hisCtcScore = hisCtcScores[i]
        for j = 1, beamSize+1 do -- iterate curToken (beamSize+1 in case of blankToken)
            local idx = maxIndices[j]
            local curToken = idx - 1
            local ctcScore = ctcScores[idx]
            local newCtcScore = hisCtcScore + ctcScore
            if not (curToken == self.blankToken) then
                local sentenceSpace, sentence, score =  getScore(curToken, hisSentenceSpace, hisSentence)
                local pos = (j - 1) * beamSize + i
                scores[pos] = newCtcScore + lmAlpha * score
                sentences[pos] = sentence
                sentencesSpace[pos] = sentenceSpace
                newCtcScores[pos] = newCtcScore
            else
                local pos = (j - 1) * beamSize + i
                scores[pos] = -math.huge
                sentences[pos] = hisSentence
                sentencesSpace[pos] = hisSentenceSpace
                newCtcScores[pos] = -math.huge
            end
        end
    end
    
    local beamSentences, beamSentencesSpace, beamCtcScores = {}, {}, {}
    scores = torch.Tensor(scores)
    local beamScores, sortedIndices = torch.topk(scores, beamSize, 1, true)
    sortedIndices = sortedIndices:float():squeeze()
    for i = 1, beamSize do
        local idx = sortedIndices[i]
        table.insert(beamSentences, sentences[idx])
        table.insert(beamSentencesSpace, sentencesSpace[idx])
        table.insert(beamCtcScores, newCtcScores[idx])
    end
    
    sentencesSpace, token2alphabet, sentences, maxIndices, sortedIndices, newCtcScores = nil, nil, nil, nil, nil, nil
    collectgarbage("collect")
    return beamSentencesSpace, beamSentences, beamCtcScores, beamScores
end