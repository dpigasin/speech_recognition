import os
import re
from argparse import ArgumentParser
from time import perf_counter

import numpy as np
from openvino.runtime import Core

if os.name == 'nt':
    os.environ['PATH'] += os.pathsep + 'ffmpeg'
from pydub import AudioSegment

from Wav2Vec import Wav2Vec


def parse_diagnosis(text):
    key_phrases = ['you have been diagnosed with', 'you have', 'diagnosis is', 'result of the examination is']
    pattern = f'({"|".join(key_phrases)})' + r'( a)? ([a-z]+)'
    return re.findall(pattern, text)[0][-1]


def read_wav(file_path):
    audio = AudioSegment.from_wav(file_path)
    audio = audio.set_sample_width(2).set_channels(1).set_frame_rate(16000)
    audio = np.array(audio.get_array_of_samples()) / np.iinfo(np.int16).max
    return np.expand_dims(audio, 0)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', required=True,
                        help='Required. Path to an audio .wav file.')
    parser.add_argument('-m', '--model', default='models/public/wav2vec2-base/FP32/wav2vec2-base.xml',
                        help='Optional. Path to an .xml file with a trained model.')
    parser.add_argument('-d', '--device', default='CPU',
                        help='Optional. Specify the target device to infer on: CPU, GPU, HDDL, MYRIAD or HETERO.')
    return parser.parse_args()


def main():
    args = parse_args()

    audio = read_wav(args.input)

    core = Core()
    model = Wav2Vec(core, args.model, audio.shape, args.device)

    start_time = perf_counter()

    text = model.recognize(audio)
    print(f'Full text: {text}')
    print(f'Diagnosis: {parse_diagnosis(text)}')

    print(f'Time: {(perf_counter() - start_time) * 1000:.2f} ms')


if __name__ == '__main__':
    main()
