from itertools import groupby

import numpy as np
from openvino.runtime import PartialShape


class Wav2Vec:
    VOCAB = dict(enumerate([
        '<pad>', '<s>', '</s>', '<unk>', '|',
        'e', 't', 'a', 'o', 'n', 'i', 'h', 's', 'r', 'd', 'l', 'u', 'm',
        'w', 'c', 'f', 'g', 'y', 'p', 'b', 'v', 'k', "'", 'x', 'j', 'q', 'z',
    ]))

    WORD_SEP = '|'
    PAD_TOKEN = '<pad>'  # noqa: S105

    def __init__(self, core, model_path, input_shape=-1, device='CPU'):
        model = core.read_model(model_path)
        self.input_name = model.inputs[0].get_any_name()

        if input_shape != -1:
            model.reshape({self.input_name: PartialShape(input_shape)})
        elif not model.is_dynamic():
            model.reshape({self.input_name: PartialShape((-1, -1))})

        compiled_model = core.compile_model(model, device)
        self.infer_request = compiled_model.create_infer_request()
        self.output_tensor = compiled_model.outputs[0]

    @staticmethod
    def _preprocess(audio):
        return (audio - np.mean(audio)) / (np.std(audio) + 1e-15)

    @staticmethod
    def _postprocess(token_probs):
        token_ids = np.squeeze(np.argmax(token_probs, -1))
        tokens = [Wav2Vec.VOCAB[idx] for idx in token_ids]
        tokens = [token_group[0] for token_group in groupby(tokens)]
        tokens = [token for token in tokens if token != Wav2Vec.PAD_TOKEN]
        return ''.join(tokens).replace(Wav2Vec.WORD_SEP, ' ').strip()

    def recognize(self, audio):
        return self._postprocess(self._infer(self._preprocess(audio)))

    def _infer(self, audio):
        return self.infer_request.infer({self.input_name: audio})[self.output_tensor]
