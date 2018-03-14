require 'mapper.Encoder'
require 'mapper.Ngram'
local utils = require 'datasets.Utils'
local paths = require 'paths'
-- construct an object to deal with the mapping
local mapper = torch.class('Mapper')

function mapper:__init(opt)
    self.GPU = opt.nGPU > 0
    
    self.encoder = Encoder(opt)
    self.alphabet2token = self.encoder.alphabet2token
    self.token2alphabet = self.encoder.token2alphabet
    self.dictionarySize = self.encoder.dictionarySize
    
    if opt.lmAlpha > 0 then
        local lmPath = opt.languageModelPath
        assert(paths.filep(lmPath), lmPath ..' not found')
        self.ngramLM = Ngram(opt, self.token2alphabet, self.dictionarySize, self.alphabet2token['$'])
        self.beamSize = opt.beamSize
    end
end

function mapper:encodeString(line)
    return self.encoder:encodeString(line)
end

function mapper:decodeOutput(prediction)
    --[[
        Turns the prediction tensor into a string
        prediction: seqLength * features
        NOTE:
            to compute WER we strip the begining and ending spaces
    --]]
    local sentence = ""
    local blankToken = self.alphabet2token['$']
    local preToken = blankToken
    local seqLength = prediction:size(1)
    -- without language model
    if (not self.ngramLM) or seqLength == 1 then
        -- The prediction is a sequence of likelihood vectors
        local _, maxIndices = torch.max(prediction, 2)
        maxIndices = maxIndices:float():squeeze()

        for i = 1, seqLength do
            local token = maxIndices[i] - 1 -- token starts from 0
            -- add token if it's not blank, and is not the same as pre_token
            if token ~= blankToken and token ~= preToken then
                sentence = sentence .. self.token2alphabet[token]
            end
            preToken = token
        end
    -- with language model and beamsearch
    elseif self.beamSize > 1 then
        if self.beamSize > 20 then
            local beamSize = self.beamSize
            local beamCtcProbs, beamCtcIdx = torch.topk(prediction, beamSize, 2, true)
            beamCtcIdx = beamCtcIdx - 1  -- token starts from 0
            local beamProbs = beamCtcProbs[1]
            local beamCtcTokens = beamCtcIdx[1]:view(beamSize, 1):totable()
            for i = 2, seqLength do
                local beamLocalProbs = torch.Tensor(beamSize, beamSize)
                if self.GPU then beamLocalProbs = beamLocalProbs:cuda() end
                for j = 1, beamSize do
                    beamLocalProbs[j] = beamCtcProbs[i] + beamProbs[j]
                end
                beamLocalProbs = beamLocalProbs:view(beamSize*beamSize)
                local sortedProbs, sortedIdx = torch.topk(beamLocalProbs, beamSize, 1, true)
                beamProbs = sortedProbs
                sortedIdx = sortedIdx:float():squeeze()
                local beamNewCtcTokens = {}
                for j = 1, beamSize do
                    local idx = sortedIdx[j]
                    local beamId = math.ceil(idx/beamSize)
                    local token = beamCtcIdx[i][(idx - 1) % beamSize + 1]   -- token starts from 0
                    local newTokens = { table.unpack(beamCtcTokens[beamId]) }
                    table.insert(newTokens, token)
                    table.insert(beamNewCtcTokens, newTokens)
                end
                beamCtcTokens = beamNewCtcTokens
            end
            sentence, _ = self.ngramLM:getMaxSentence(beamCtcTokens, beamProbs)
            beamCtcTokens, beamProbs = nil, nil
            collectgarbage("collect")
        else
            local beamSize = self.beamSize
            local beamSentences, beamSentencesSpace = {}, {}
            local beamCtcScores = torch.Tensor(beamSize):zero()
            if self.GPU then beamCtcScores = beamCtcScores:cuda() end
            local beamScores = beamCtcScores:clone()
            
            local _, maxIndices = torch.max(prediction, 2)
            maxIndices = maxIndices:float():squeeze() - 1 -- token starts from 0
            for i = 1, seqLength do
                local token = maxIndices[i]
                -- add token if it's not blank, and is not the same as pre_token
                if token ~= blankToken and token ~= preToken then
                    if #beamSentences>0 then
                        beamSentencesSpace, beamSentences, beamCtcScores, beamScores = self.ngramLM:getBeamPredictions(beamSentencesSpace, beamSentences, prediction[i], beamCtcScores, beamSize)
                    else
                        local beamCtcProbs, beamCtcIdx = torch.topk(prediction[i], beamSize+1, 1, true)  --(beamSize+1 in case of blankToken)
                        beamCtcIdx = beamCtcIdx - 1  -- token starts from 0
                        local counter, lowestProb, lowestIdx = 0, math.huge, 0
                        for i = 1, beamSize+1 do 
                            local ctcToken = beamCtcIdx[i]
                            if not (ctcToken == blankToken) then
                                local character = self.token2alphabet[ctcToken]
                                local ctcProb = beamCtcProbs[i]
                                if counter == beamSize then
                                    if ctcProb > lowestProb then
                                        beamSentences[lowestIdx] = character
                                        beamSentencesSpace[lowestIdx] = character
                                        beamCtcScores[lowestIdx] = ctcProb
                                    end
                                else
                                  counter = counter + 1
                                  table.insert(beamSentences, character)
                                  table.insert(beamSentencesSpace, character)
                                  beamCtcScores[counter] = ctcProb
                                  if ctcProb < lowestProb then 
                                      lowestProb = ctcProb
                                      lowestIdx = counter
                                  end
                               end
                            end
                        end
                    end
                end
                preToken = token
            end
            local _, maxIdx = torch.max(beamScores, 1)
            sentence = beamSentences[maxIdx[1]]
            beamSentences, beamSentencesSpace, beamCtcScores, beamScores, maxIdx = nil, nil, nil, nil, nil
            collectgarbage("collect")
        end
    -- with language model and without beamsearch
    else
        local beamSize = 10
        local sentenceSpace = ""
        -- The prediction is a sequence of likelihood vectors
        local _, maxIndices = torch.max(prediction, 2)
        maxIndices = maxIndices:float():squeeze() - 1 -- token starts from 0
        
        for i = 1, seqLength do
            local token = maxIndices[i]
            -- add token if it's not blank, and is not the same as pre_token
            if token ~= blankToken and token ~= preToken then
                if #sentence>0 then
                    sentenceSpace, sentence, _ = self.ngramLM:getBeamMaxPrediction(sentenceSpace, sentence, prediction[i], beamSize)
                else
                    sentence = self.token2alphabet[token]
                    sentenceSpace = sentence
                end
            end
            preToken = token
        end
    end
    collectgarbage("collect")
    return sentence
end

function Mapper:closeDecoder()
    if self.beamSize then
        self.ngramLM:closeServer()
    end
end