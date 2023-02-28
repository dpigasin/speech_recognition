# Recognition of a medical diagnosis from speech

The program parses a medical diagnosis from speech after key phrases using a regular expression.  
Text-to-speech conversion is performed using a [Wav2Vec model](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/wav2vec2-base) that runs on OpenVINO.  
The model expects 16-bit, 16 kHz, mono-channel WAVE audio as input data, but you can use any .wav file.  
It will be converted to the desired format in runtime (ffmpeg is needed). 

## Usage

```
python main.py -h
usage: main.py [-h] -i INPUT [-m MODEL] [-d DEVICE]

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Required. Path to an audio .wav file.
  -m MODEL, --model MODEL
                        Optional. Path to an .xml file with a trained model.
  -d DEVICE, --device DEVICE
                        Optional. Specify the target device to infer on: CPU, GPU, HDDL, MYRIAD or HETERO.
```

## Examples

```
python main.py -i input_data\gastritis.wav
Full text: you have been diagnosed with gastritis so you need to stick to a diet
Diagnosis: gastritis
Time: 646.77 ms

python main.py -i input_data\cancer.wav
Full text: your diagnosis is cancer i'm sorry
Diagnosis: cancer
Time: 486.73 ms

python main.py -i input_data\salmonellosis.wav
Full text: the result of the examination is salmonelosis he will have to go to the hospital for a few weeks
Diagnosis: salmonelosis
Time: 804.45 ms
```
