from enum import Enum
from abc import ABCMeta, abstractmethod
import onnx
import os
import urllib.request as urllib


class ModelType(Enum):
    ResNet50 = "resnet50"
    VGG19 = "vgg19"


class ModelLoaderFactory:
    @classmethod
    def get_loader(cls, model_type, model_folder):
        if model_type == ModelType.ResNet50:
            return ResNet50OnnxLoader(model_folder)
        elif model_type == ModelType.VGG19:
            return VGG19OnnxLoader(model_folder)
        else:
            raise NotImplementedError("{} model is not supported!".format(model_type))


class ModelLoader:
    __metaclass__ = ABCMeta

    def __init__(self, model_folder, model_name):
        self._model_folder = model_folder
        self._model_name = model_name

    @abstractmethod
    def _download(self):
        pass

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def exist(self):
        pass

    @abstractmethod
    def model_path(self):
        pass


class OnnxLoader(ModelLoader):
    def load(self):
        return onnx.load_model(self.model_path())

    def exist(self):
        return os.path.exists(self.model_path())

    def model_path(self):
        return os.path.join(self._model_folder, self._model_name + ".onnx")

    def _download(self):
        raise NotImplementedError("this function is not implemented.")


class RemoteOnnxLoader(OnnxLoader):
    def __init__(self, model_folder, model_name, onnx_url):
        super(RemoteOnnxLoader, self).__init__(model_folder, model_name)
        self._url = onnx_url

    def _download(self):
        ModelUtil.download(self._url, self.model_path())

    def load(self):
        if not self.exist():
            self._download()
        return onnx.load_model(self.model_path())


class ResNet50OnnxLoader(RemoteOnnxLoader):
    def __init__(self, model_folder):
        super(ResNet50OnnxLoader, self).__init__(
            model_folder=model_folder,
            model_name=ModelType.ResNet50.value,
            onnx_url="https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet50v2/resnet50v2.onnx"
        )


class VGG19OnnxLoader(RemoteOnnxLoader):
    def __init__(self, model_folder):
        super(VGG19OnnxLoader, self).__init__(
            model_folder=model_folder,
            model_name=ModelType.VGG19.value,
            onnx_url="https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg19/vgg19.onnx"
        )


class ModelUtil:
    @classmethod
    def download(cls, url, path, overwrite=False):
        if os.path.isfile(path) and not overwrite:
            print('File {} existed, skip.'.format(path))
            return

        print('Downloading from url {} to {}'.format(url, path))
        try:
            urllib.request.urlretrieve(url, path)
        except:
            urllib.urlretrieve(url, path)
