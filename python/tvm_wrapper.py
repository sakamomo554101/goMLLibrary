import model_loader
import nnvm
import tvm
import os
import json


class TvmCompileParameter:
    def __init__(self, model_root_path, model_type, output_folder, shape_dict, data_dict="float32", target="llvm", opt_level=3, ctx=tvm.cpu(0), debug=True):
        self.model_root_path = model_root_path
        self.model_type = model_type
        self.output_folder = output_folder
        self.target = target
        self.opt_level = opt_level
        self.data_dict = data_dict
        self.shape_dict = shape_dict
        self.ctx = ctx
        self.debug = debug


class TvmWrapper:
    def __init__(self, param):
        self.__param = param
        self.__model = None

        # フォルダが存在しない場合は作成する
        os.makedirs(self.__param.model_root_path, exist_ok=True)
        os.makedirs(self.__param.output_folder, exist_ok=True)

    def setup(self):
        loader = model_loader.ModelLoaderFactory.get_loader(
            model_folder=self.__param.model_root_path,
            model_type=self.__param.model_type
        )
        self.__model = loader.load()

    # TODO : setup前に実行した場合に例外を発生させる
    def compile(self):
        # モデルからsymbol, parameterを取得
        # TODO : Darknetの対応が必要
        sym, params = nnvm.frontend.from_onnx(self.__model)

        # symbol, parameterからビルドを実施
        # TODO : Relayの対応
        with nnvm.compiler.build_config(opt_level=self.__param.opt_level):
            # build
            graph, lib, params = nnvm.compiler.build(
                graph=sym,
                target=self.__param.target,
                shape=self.__param.shape_dict,
                dtype=self.__param.data_dict,
                params=params
            )

            # TVM用のモデルデータをexport
            self.__export_model(graph, lib, params)

            # グラフを作成
            if self._ir_config.debug:
                from tvm.contrib.debugger import debug_runtime as graph_runtime
            else:
                from tvm.contrib import graph_runtime
            graph, params = graph_runtime.create(graph, lib, self.__param.ctx)
            input_name = self.__model.graph.input[0].name
            return graph, params, input_name

    def __export_model(self, graph, lib, params):
        # ライブラリをexport
        model_name = self.__param.model_type.value
        lib_name = model_name + ".so"
        lib.export_library(os.path.join(self.__param.output_folder, lib_name))

        # グラフ情報をファイルに書き出す
        graph_name = model_name + ".json"
        graph_json = graph.json()
        with open(os.path.join(self.__param.output_folder, graph_name), "w") as f:
            f.write(graph_json)

        # パラメーター情報をファイルに書き出す
        param_name = model_name + ".params"
        param_bytes = nnvm.compiler.save_param_dict(params)
        with open(os.path.join(self.__param.output_folder, param_name), "wb") as f:
            f.write(param_bytes)

    # TODO : compile前に実行した場合に例外を発生させる
    def execute(self, graph, params, input_name, x):
        graph.set_input(input_name, tvm.nd.array(x.astype("float32")))
        graph.set_input(params)
        graph.run()
        return graph.get_output(0)


if __name__ == "__main__":
    data_dict = {"reshape_attr_tensor430": "int64"}
    shape_dict = {"data": [1, 3, 224, 224]}
    model_root_path = "model"
    output_folder = "output"
    compile_param = TvmCompileParameter(
        model_root_path=model_root_path,
        model_type=model_loader.ModelType.ResNet50,
        output_folder=output_folder,
        shape_dict=shape_dict,
        data_dict=data_dict
    )

    wrapper = TvmWrapper(compile_param)
    wrapper.setup()
    wrapper.compile()
