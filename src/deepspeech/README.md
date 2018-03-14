Chinese DeepSpeech 2 in Torch
=============================

This is the Chinsese-only speech recognition model implementation based on [DeepSpeech2](http://arxiv.org/pdf/1512.02595v1.pdf) architecture using the Torch7 library, trained with the CTC activation function. 

The code is modified from [deepspeech.torch](https://github.com/SeanNaren/deepspeech.torch), and it supports N-gram language model embedding.

Installation
------------

### GPU Support

To get GPU Support, you need to install [CUDA 8.0](https://developer.nvidia.com/cuda-downloads) and [cuDNN](https://developer.nvidia.com/cudnn) >5.0. Please follow the instructions provided on the official NVIDIA website.

### Torch 7

[Torch](http://torch.ch/) official provides a simple installation process for Torch on Mac OS X and Ubuntu 12+:
```bash
# in a terminal, run the commands WITHOUT sudo
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch; bash install-deps;
./install.sh
# On Linux with bash
source ~/.bashrc
# On OSX or in Linux with none of the above.
source ~/.profile
```

### Other Dependencies

Please run the following commands in order.
```bash
luarocks install tds
luarocks install torch
luarocks install nn
luarocks install nnx
luarocks install dpnn
luarocks install rnn
luarocks install optim

luarocks install cutorch
luarocks install cunn
luarocks install cunnx
luarocks install cudnn

luarocks install nngraph
luarocks install parallel
luarocks install threads
luarocks install xlua

luarocks install http://raw.githubusercontent.com/baidu-research/warp-ctc/master/torch_binding/rocks/warp-ctc-scm-1.rockspec

sudo apt-get install libfftw3-dev
sudo apt-get install sox libsox-dev libsox-fmt-all
luarocks install https://raw.githubusercontent.com/soumith/lua---audio/master/audio-0.1-0.rockspec

git clone https://github.com/LMDB/lmdb
cd lmdb/libraries/liblmdb/
make
sudo make install

git clone https://github.com/eladhoffer/lmdb.torch
cd lmdb.torch
luarocks make
```

Data Preparation
----------------

Here we take [THCHS30 Chinsese Audio Dataset](http://data.cslt.org/thchs30/README.html) as example.

Please run the following commands.

```bash
cd src/deepspeech_torch/datasets

# download thchs30 dataset
mkdir -p thchs30/thchs30 ; cd thchs30/thchs30
for part in wav doc lm ; do
    wget "http://data.cslt.org/thchs30/zip/$part.tgz"
    tar -xzvf "$path.tgz"
done

# prepare dataset
cd ../../
# origin dataset is incomplete, we need to generate training data with noise by hand
th GenTHchs30Noise.lua -reproduce # reproduce option is to reproduce test/cv-noise data since our memroy is not enough

# organize thchs30 dataset
th FormatTHchs30.lua -noise # noise option is to include those noise data

# resplit dataset, since origin train set only has 1600 characters while there are 2800+ in whole dataset
th ResplitDataset.lua -rootPath thchs30/thchs30_noise_dataset
mv thchs30/thchs30_noise_dataset_new thchs30/thchs30_dataset

# make LMDB
th MakeLMDB.lua -rootPath thchs30/thchs30_dataset -lmdbPath thchs30/thchs30_noise_lmdb -windowSize 0.02 -stride 0.01 -sampleRate 16000

# create dictionary
th CreateChsDictionary.lua -textDir thchs30/thchs30_dataset/train -countThreshold 2
```

### Adding new dataset

If you want to adding new dataset, please organize your dataset as follows
> | your\_dataset\_dir\_name
>
> &nbsp;&nbsp;&nbsp;&nbsp;|-train
>
> &nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- A1.wav
>
> &nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- A1.txt
>
> &nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|- ...
>
> &nbsp;&nbsp;&nbsp;&nbsp;|-cv
>
> &nbsp;&nbsp;&nbsp;&nbsp;|-test

Then you can run `MakeLMDB.lua -rootPath PATH_TO_YOUR_DATASET -lmdbPath PATH_TO_YOUR_LMDB_DATASET` as mentioned above.

Training and Validation
-------

To execute (based on the parameters above, you may need to change it for your run paths, GPUs etc) on THCHS30, there are two commands, one for a lightweight model which is easier to train, and one for a pure DeepSpeech model:

### Lightweight
```bash
th Train.lua -hiddenSize 600 -rnnType lstm -nbOfHiddenLayers 5 -epochSave 1 -cudnnFastest -permuteBatch -trainingSetLMDBPath datasets/thchs30/thchs30_lmdb/train/ -validationSetLMDBPath datasets/thchs30/thchs30_lmdb/cv/ -dictionaryPath datasets/thchs30/dictionary

th Test.lua -loadPath PATH_TO_YOUR_T7_MODEL -trainingSetLMDBPath datasets/thchs30/thchs30_lmdb/train/ -validationSetLMDBPath datasets/thchs30/thchs30_lmdb/test/ -dictionaryPath datasets/thchs30/dictionary
```

### DeepSpeech
```bash
th Train.lua -epochSave 1 -cudnnFastest -permuteBatch -trainingSetLMDBPath datasets/thchs30/thchs30_lmdb/train/ -validationSetLMDBPath datasets/thchs30/thchs30_lmdb/cv/ -dictionaryPath datasets/thchs30/dictionary

th Test.lua -loadPath PATH_TO_YOUR_T7_MODEL -trainingSetLMDBPath datasets/thchs30/thchs30_lmdb/train/ -validationSetLMDBPath datasets/thchs30/thchs30_lmdb/test/ -dictionaryPath datasets/thchs30/dictionary
```

Evaluation
----------

Please run the following commands.
```bash
th Predict.lua -modelPath PATH_TO_YOUR_T7_MODEL -audioPath PATH_TO_AUDIO_FILE -dictionaryPath datasets/thchs30/dictionary
```

Language Model
--------------

### Generate Language Model

If you want to combine language model while decoding the predictions, first you need to generate Chinese Character Based N-gram language model.

 - If you do not have Chinese text to generate model, you can use transcripts of THCHS30 dataset. Please run the following command to generate the Chinese text.

    ```bash
    cd src/deepspeech_torch/datasets
    th CreateChs30Ngram.lua -textDir thchs30/thchs30_dataset -outputTextPath thchs30/ngram.txt
    ```
 - If you do have some Chinese text files, please merge them into one file and delete all punctuations. You can run the script `src/deepspeech_torch/datasets/ProcessChineseText.py --in-file <input text file path> --out-file <output text file path>` to help you format your single Chinese text file. In case that there might be some characters in the THCHS30 transcripts not showing in your own text files, you need to run the following command to merge THCHS30 transcripts and your processed text file.
    ```bash
    cd cd src/deepspeech_torch/datasets
    th CreateChsNgram.lua -textDir thchs30/thchs30_dataset -otherTextPath PATH_TO_YOUR_OWN_TEXT_FILE -outputTextPath thchs30/ngram.txt
    ```
 - You **NEED** other softwares, such as [mitlm](https://github.com/mitlm/mitlm), [kenlm](https://github.com/kpu/kenlm) to generate n-gram language model from Chinese Text file. In the scprit `src/deepspeech_torch/datasets/CreateChsNgram.lua`, we use kenlm as default. If you have installed kenlm, please run the following commands instead of those above.
    ```bash
    cd cd src/deepspeech_torch/datasets
    # generate 5-gram language model
    th CreateTHchs30Ngram.lua -textDir thchs30/thchs30_dataset -otherTextPath PATH_TO_YOUR_OWN_TEXT_FILE -outputTextPath thchs30/ngram.txt -trainLanguageModel -kenlmPath PATH_TO_YOUR_KENLM_BIN_DIR -n 5 -languageModelPath thchs30/chs5gram.lm
    ```

### Evaluation with Language Model and Beamsearch

For scripts `src/deepspeech_torch/Train.lua`, `src/deepspeech_torch/Test.lua` and `src/deepspeech_torch/Predict.lua`, you can use options shown below to evaluate cv/test/own dataset.

```bash
# example
th Test.lua -languageModelPath ./datasets/thchs30/ngram.lm -lmAlpha 5 -beamSize 200 -cudnnFastest
```
