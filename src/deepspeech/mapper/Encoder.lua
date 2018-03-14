local utils = require 'datasets.Utils'
local paths = require 'paths'
-- construct an object to deal with the mapping
local encoder = torch.class('Encoder')

function encoder:__init(opt)
    local dictPath = opt.dictionaryPath
    assert(paths.filep(dictPath), dictPath ..' not found')
    
    -- make maps
    self.alphabet2token = {}
    self.token2alphabet = {}
    local cnt = 0
    for line in io.lines(dictPath) do
        self.alphabet2token[line] = cnt  -- ctc token starts from 0
        self.token2alphabet[cnt] = line
        cnt = cnt + 1
    end
    
    self.dictionarySize = cnt
end

function encoder:encodeString(line)
    line = string.lower(line)
    local label = {}
    local function getTokens(character)
        local token = self.alphabet2token[character]
        if token then
            table.insert(label, token)
        else
            --print('warning: ' .. character .. ' is not in dictionary - using * instead')
            table.insert(label, self.alphabet2token['*'])
        end
     end
    utils.chsFilter(line, getTokens)
    return label
end