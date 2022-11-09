from abc import ABC, abstractmethod
import functools
from operator import mul

import json
import requests
import numpy as np
import stackprinter

from tritonclient.utils import InferenceServerException

from app.config import SERVING_HOST, SERVING_TRITON_PROTOCOL, TORCHSERVE_MANAGEMENT_URL, TORCHSERVE_INFERENCE_URL

if SERVING_TRITON_PROTOCOL == 'grpc':
    import tritonclient.grpc as triton_lib
elif SERVING_TRITON_PROTOCOL == 'http':
    import tritonclient.http as triton_lib
else:
    raise TypeError(f'Triton client type {SERVING_TRITON_PROTOCOL} not supported.')

from app.error import CustomError


class Serving(ABC):
    """
    Serving base class.
    """

    @abstractmethod
    def health(self, *args, **kwargs) -> bool:
        """Check that the server is healthy and ready"""
        pass

    @abstractmethod
    def load(self, *args, **kwargs) -> None:
        """Load a model to server"""
        pass

    @abstractmethod
    def unload(self, *args, **kwargs) -> None:
        """Unload a model from server"""
        pass

    @abstractmethod
    def models(self, *args, **kwargs) -> list:
        """List models live in the server"""
        pass

    @abstractmethod
    def exists(self, *args, **kwargs) -> bool:
        """Check if a model is ready for inference request"""
        pass


class NvidiaServing(Serving, ABC):
    """Nvidia Serving Client base class"""

    @staticmethod
    def _infer_input_shape(input_arr) -> list:
        return list(input_arr.shape)

    @abstractmethod
    def classify(self, *args, **kwargs):
        pass

    @abstractmethod
    def classify_paddle(self, *args, **kwargs):
        pass

    @abstractmethod
    def crop_dhsegment(self, *args, **kwargs):
        pass

    @abstractmethod
    def detect_craft(self, *args, **kwargs):
        pass

    @abstractmethod
    def detect_effdet(self, *args, **kwargs):
        pass

    @abstractmethod
    def detect_yolov7(self, *args, **kwargs):
        pass

    @abstractmethod
    def read_attention(self, *args, **kwargs):
        pass

    @abstractmethod
    def read_deeptext(self, *args, **kwargs):
        pass

    @abstractmethod
    def reader_deeptext_encoder(self, *args, **kwargs):
        pass

    @abstractmethod
    def reader_deeptext_decoder(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_entity(self, *args, **kwargs):
        pass


class TritonServing(NvidiaServing, ABC):
    """
    TritonServer Client base class.
    """

    @classmethod
    def _get_triton_client(cls):
        try:
            triton_client = triton_lib.InferenceServerClient(url=SERVING_HOST)
        except InferenceServerException:
            stackprinter.show()
            raise Exception("Can not create Inference Server Client, plz check Serving or network")

        return triton_client

    @classmethod
    def _set_input(cls, input_obj: triton_lib.InferInput, arr: np.ndarray) -> triton_lib.InferInput:
        input_obj.set_data_from_numpy(arr)
        return input_obj

    @classmethod
    def _infer(cls, model_name, inputs, outputs):
        triton_client = cls._get_triton_client()

        try:
            results = triton_client.infer(
                model_name=model_name,
                inputs=inputs,
                outputs=outputs
            )
        except InferenceServerException:
            stackprinter.show()
            raise CustomError(error_code="P01", message="Inference failed, check Serving or model repository.")

        return results

    @classmethod
    def classify(cls, model_name, img, input_shape=None):
        """
        Perform inference request using TritonServer
        :param model_name:
        :param img:
        :param input_shape: str or shape list
        :return:
        """
        inputs = []
        outputs = []

        if not input_shape:
            input_shape = cls._infer_input_shape(img)

        inputs.append(triton_lib.InferInput('input', input_shape, "FP32"))
        inputs[0] = cls._set_input(inputs[0], img)

        outputs.append(triton_lib.InferRequestedOutput('output'))

        out = cls._infer(model_name=model_name,
                         inputs=inputs,
                         outputs=outputs)

        return out.as_numpy('output')

    @classmethod
    def classify_paddle(cls, model_name, img, input_shape=None):
        """
        Perform inference request using TritonServer
        :param model_name:
        :param img:
        :param input_shape: str or shape list
        :return:
        """
        inputs = []
        outputs = []

        if not input_shape:
            input_shape = cls._infer_input_shape(img)

        inputs.append(triton_lib.InferInput('x', input_shape, "FP32"))
        inputs[0] = cls._set_input(inputs[0], img)

        outputs.append(triton_lib.InferRequestedOutput('output'))

        out = cls._infer(model_name=model_name,
                         inputs=inputs,
                         outputs=outputs)

        return out.as_numpy('output')

    @classmethod
    def crop_dhsegment(cls, model_name, img, input_shape=None):
        inputs = []
        outputs = []

        if not input_shape:
            input_shape = cls._infer_input_shape(img)

        inputs.append(triton_lib.InferInput('image', input_shape, "FP32"))
        inputs[0] = cls._set_input(inputs[0], img)

        outputs.append(triton_lib.InferRequestedOutput('labels'))
        outputs.append(triton_lib.InferRequestedOutput('probs'))

        out = cls._infer(model_name=model_name,
                         inputs=inputs,
                         outputs=outputs)

        new_labels = out.as_numpy('labels')
        new_probs = out.as_numpy('probs')
        return new_labels, new_probs

    @classmethod
    def detect_craft(cls, model_name, x):
        # craft
        inputs = []
        outputs = []
        input_shape = cls._infer_input_shape(x)

        inputs.append(triton_lib.InferInput('input.1', input_shape, "FP32"))
        inputs[0] = cls._set_input(inputs[0], x)

        outputs.append(triton_lib.InferRequestedOutput('285'))
        outputs.append(triton_lib.InferRequestedOutput('275'))

        out_craft = cls._infer(model_name=model_name,
                               inputs=inputs,
                               outputs=outputs)
        o_285 = out_craft.as_numpy('285')
        o_275 = out_craft.as_numpy('275')

        # craft refiner
        inputs = []
        outputs = []
        inputs.append(triton_lib.InferInput('0', cls._infer_input_shape(o_285), "FP32"))
        inputs.append(triton_lib.InferInput('1', cls._infer_input_shape(o_275), "FP32"))
        inputs[0] = cls._set_input(inputs[0], o_285)
        inputs[1] = cls._set_input(inputs[1], o_275)

        outputs.append(triton_lib.InferRequestedOutput('129'))
        out_refiner = cls._infer(model_name=model_name + '-refiner',
                                 inputs=inputs,
                                 outputs=outputs)

        return o_285, out_refiner.as_numpy('129')

    @classmethod
    def detect_effdet(cls, model_name, anchor, img, input_1_shape=None, input_4_shape=None):
        """
        Detect image objects using EfficientDet model.
        The input img shape is [-1, {size}, {size}, 3] where size taken from [512, 640, 768, 896, 1024, 1280, 1408]
        by params "phi" in higher level code and relate to app config.
        So we use current input to infer the input shape.
        :param model_name: The name of model has trained.
        :param anchor:
        :param img: input img
        :param input_1_shape:
        :param input_4_shape:
        :return: boxes, scores, labels of objects found on input img
        """

        inputs = []
        outputs = []

        if not input_1_shape:
            input_1_shape = cls._infer_input_shape(img)  # already have batch dim
        if not input_4_shape:
            input_4_shape = cls._infer_input_shape(anchor)

        inputs.append(triton_lib.InferInput('input_1', input_1_shape, "FP32"))
        inputs.append(triton_lib.InferInput('input_4', input_4_shape, "FP32"))
        inputs[0] = cls._set_input(inputs[0], img)
        inputs[1] = cls._set_input(inputs[1], anchor)

        outputs.append(triton_lib.InferRequestedOutput('boxes'))
        outputs.append(triton_lib.InferRequestedOutput('scores'))
        outputs.append(triton_lib.InferRequestedOutput('labels'))

        out = cls._infer(model_name=model_name,
                         inputs=inputs,
                         outputs=outputs)

        return out.as_numpy("boxes"), out.as_numpy("scores"), out.as_numpy("labels")

    @classmethod
    def detect_yolov7(cls, model_name, img, input_shape=None):
        """
        Detect image objects using EfficientDet model.
        The input img shape is [-1, {size}, {size}, 3] where size taken from [512, 640, 768, 896, 1024, 1280, 1408]
        by params "phi" in higher level code and relate to app config.
        So we use current input to infer the input shape.
        :param model_name: The name of model has trained.
        :param img: input img
        :param input_1_shape:
        :return: boxes, scores, labels of objects found on input img
        """

        inputs = []
        outputs = []

        if not input_shape:
            input_shape = cls._infer_input_shape(img)  # already have batch dim

        inputs.append(triton_lib.InferInput('images', input_shape, "FP32"))
        inputs[0] = cls._set_input(inputs[0], img.astype(np.float32))

        outputs.append(triton_lib.InferRequestedOutput('output'))

        out = cls._infer(model_name=model_name,
                         inputs=inputs,
                         outputs=outputs)

        return out.as_numpy('output')

    @classmethod
    def read_attention(cls, model_name, img, input_shape=None):
        """
        :param model_name:
        :param img: np array with shape of [batch, *img_shape]
        :param input_shape:
        :return:
        """

        inputs = []
        outputs = []

        if not input_shape:
            input_shape = [1] + cls._infer_input_shape(img)

        inputs.append(triton_lib.InferInput('input', input_shape, "FP32"))
        inputs[0] = cls._set_input(inputs[0], img)

        outputs.append(triton_lib.InferRequestedOutput('output'))
        outputs.append(triton_lib.InferRequestedOutput('prob'))

        out = cls._infer(model_name=model_name,
                         inputs=inputs,
                         outputs=outputs)

        text = out.as_numpy('output')[0]
        text = text.decode('utf-8').strip(u'\u2591')

        probs = out.as_numpy('prob')[0].tolist()
        prob = functools.reduce(mul, probs[:len(text)])

        return text, prob

    @classmethod
    def read_deeptext(cls, model_name, img, input_shape=None):

        if not input_shape:
            input_shape = cls._infer_input_shape(img)  # already have batch dim

        inputs = []
        outputs = []
        inputs.append(triton_lib.InferInput('input', input_shape, "FP32"))
        inputs[0] = cls._set_input(inputs[0], img)

        outputs.append(triton_lib.InferRequestedOutput('output'))

        out = cls._infer(model_name=model_name,
                         inputs=inputs,
                         outputs=outputs)

        # return np.expand_dims(out.as_numpy('output'), 0)
        return out.as_numpy('output')

    @classmethod
    def reader_deeptext_encoder(cls, model_name, img, input_shape=None):

        if not input_shape:
            input_shape = cls._infer_input_shape(img)

        inputs = []
        outputs = []
        inputs.append(triton_lib.InferInput('image', input_shape, "FP32"))
        inputs[0] = cls._set_input(inputs[0], img)

        outputs.append(triton_lib.InferRequestedOutput('context'))

        out = cls._infer(model_name=model_name,
                         inputs=inputs,
                         outputs=outputs)

        return out.as_numpy('context')

    @classmethod
    def reader_deeptext_decoder(cls, model_name, h0, c0, context, onehot,
                                input_h_0_shape=None,
                                input_c_0_shape=None,
                                input_context_shape=None,
                                input_onehot_shape=None):
        inputs = []
        outputs = []

        if not input_h_0_shape:
            input_h_0_shape = cls._infer_input_shape(h0)
        if not input_c_0_shape:
            input_c_0_shape = cls._infer_input_shape(c0)
        if not input_context_shape:
            input_context_shape = cls._infer_input_shape(context)
        if not input_onehot_shape:
            input_onehot_shape = cls._infer_input_shape(onehot)

        inputs.append(triton_lib.InferInput('h_0', input_h_0_shape, "FP32"))
        inputs.append(triton_lib.InferInput('c_0', input_c_0_shape, "FP32"))
        inputs.append(triton_lib.InferInput('context', input_context_shape, "FP32"))
        inputs.append(triton_lib.InferInput('onehot', input_onehot_shape, "FP32"))
        inputs[0] = cls._set_input(inputs[0], h0)
        inputs[1] = cls._set_input(inputs[1], c0)
        inputs[2] = cls._set_input(inputs[2], context)
        inputs[3] = cls._set_input(inputs[3], onehot)

        outputs.append(triton_lib.InferRequestedOutput('h_n'))
        outputs.append(triton_lib.InferRequestedOutput('c_n'))
        outputs.append(triton_lib.InferRequestedOutput('prob_step'))

        out = cls._infer(model_name=model_name,
                         inputs=inputs,
                         outputs=outputs)

        return out.as_numpy('h_n'), out.as_numpy('c_n'), out.as_numpy('prob_step')

    @classmethod
    def get_entity(cls, model_name, input_ids, attention_mask, token_type_ids,
                   input_ids_shape=None,
                   input_mask_shape=None,
                   input_token_shape=None):

        if not input_ids_shape:
            input_ids_shape = cls._infer_input_shape(input_ids)
        if not input_mask_shape:
            input_mask_shape = cls._infer_input_shape(attention_mask)
        if not input_token_shape:
            input_token_shape = cls._infer_input_shape(token_type_ids)

        inputs = []
        outputs = []

        inputs.append(triton_lib.InferInput('input_ids', input_ids_shape, "INT64"))
        inputs.append(triton_lib.InferInput('attention_mask', input_mask_shape, "INT64"))
        inputs.append(triton_lib.InferInput('token_type_ids', input_token_shape, "INT64"))
        inputs[0] = cls._set_input(inputs[0], input_ids)
        inputs[1] = cls._set_input(inputs[1], attention_mask)
        inputs[2] = cls._set_input(inputs[2], token_type_ids)

        outputs.append(triton_lib.InferRequestedOutput('pred_intent_ids'))
        outputs.append(triton_lib.InferRequestedOutput('pred_1_ids'))
        outputs.append(triton_lib.InferRequestedOutput('pred_2_ids'))

        out = cls._infer(model_name=model_name,
                         inputs=inputs,
                         outputs=outputs)

        return out.as_numpy('pred_intent_ids'), out.as_numpy('pred_1_ids'), out.as_numpy('pred_2_ids')

    @classmethod
    def health(cls):
        triton_client = cls._get_triton_client()
        return triton_client.is_server_ready()

    @classmethod
    def load(cls, model_name):
        triton_client = cls._get_triton_client()
        triton_client.load_model(model_name)

    @classmethod
    def unload(cls, model_name):
        triton_client = cls._get_triton_client()
        triton_client.unload_model(model_name)

    @classmethod
    def models(cls):
        triton_client = cls._get_triton_client()
        return triton_client.get_model_repository_index()

    @classmethod
    def exists(cls, model_name, model_version=""):
        triton_client = cls._get_triton_client()
        return triton_client.is_model_ready(model_name, model_version)


class TritonServingHTTP(TritonServing):
    pass


class TritonServingGRPC(TritonServing):
    pass


def nvidia_serving_factory() -> NvidiaServing:
    if SERVING_TRITON_PROTOCOL == 'grpc':
        return TritonServingGRPC()
    elif SERVING_TRITON_PROTOCOL == 'http':
        return TritonServingHTTP()
    else:
        raise TypeError(f'Triton client type {SERVING_TRITON_PROTOCOL} not supported.')


class TorchServing(Serving):
    @classmethod
    def health(cls):
        response = requests.get(f"{TORCHSERVE_INFERENCE_URL}/ping")
        return response.status_code == 200

    @classmethod
    def load(cls, model_name):
        response = requests.post(
            f"{TORCHSERVE_MANAGEMENT_URL}/models?url={model_name + '.mar'}&model_name={model_name}")
        if response.status_code == 200:
            requests.put(f"{TORCHSERVE_MANAGEMENT_URL}/models/{model_name}?min_worker=1")

    @classmethod
    def unload(cls, model_name):
        requests.delete(f'{TORCHSERVE_MANAGEMENT_URL}/models/{model_name}/')

    @classmethod
    def models(cls):
        response = requests.get(f"{TORCHSERVE_MANAGEMENT_URL}/models")
        if response.status_code == 200:
            return json.loads(response.content)['models']
        else:
            return []

    @classmethod
    def exists(cls, model_name):
        response = requests.get(f'{TORCHSERVE_MANAGEMENT_URL}/models/{model_name}')
        return response.status_code == 200

    @classmethod
    def get_entity_pick(cls, model_name: str, image: bytes, data_nlp: list) -> list:
        data = {'image': image, 'tsv': json.dumps(data_nlp, ensure_ascii=False)}
        response = requests.post(f'{TORCHSERVE_INFERENCE_URL}/predictions/{model_name}', data=data)
        if response.status_code == 200:
            return json.loads(response.content)
        else:
            stackprinter.show()
            raise CustomError(error_code="P01", message="Inference failed, check TorchServing or model repository.")


def torchserve_factory() -> Serving:
    return TorchServing()


class SeldonServing(Serving, ABC):
    """
    Seldon Client base class.
    """

    @classmethod
    def crop_dhsegment(cls, model_name, img, input_shape=None):
        pass

    @classmethod
    def detect_craft(cls, model_name, x):
        pass

    @classmethod
    def read_attention(cls, model_name, img, input_shape=None):
        pass

    @classmethod
    def read_deeptext_ctc(cls, model_name, img, input_shape=None):
        pass

    @classmethod
    def reader_deeptext_attn_encoder(cls, model_name, img, input_shape=None):
        pass

    @classmethod
    def reader_deeptext_attn_decoder(cls, model_name, h0, c0, context, onehot,
                                     input_h_0_shape=None,
                                     input_c_0_shape=None,
                                     input_context_shape=None,
                                     input_onehot_shape=None):
        pass

    @classmethod
    def health(cls):
        pass

    @classmethod
    def load(cls, model_name):
        pass

    @classmethod
    def unload(cls, model_name):
        pass

    @classmethod
    def models(cls):
        pass

    @classmethod
    def exists(cls, model_name, model_version=""):
        pass
