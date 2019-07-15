package model

import (
	"bytes"
	"encoding/gob"
	"errors"
	"io/ioutil"

	"github.com/goMLLibrary/core/neuralNetwork"
	"gonum.org/v1/gonum/mat"
)

// WriteNNLayers : ニューラルネットワークの情報をファイルに書き出す
func WriteNNLayers(modelPath string, nnLayers *neuralNetwork.NeuralNetworkLayers) error {
	// レイヤー情報を保存用のモデル情報に書き換える
	nnModel, err := convertNNModel(nnLayers)
	if err != nil {
		return err
	}

	// モデル情報をbyteデータに書き換え、ファイルに書き込む
	byteData, err := encodeNNModel(nnModel)
	if err != nil {
		return err
	}
	return writeModelFile(modelPath, byteData)
}

// ReadNNLayers : ニューラルネットワークの情報をファイルから取得する
func ReadNNLayers(modelPath string) (*neuralNetwork.NeuralNetworkLayers, error) {
	// ファイルからmodelのbyteデータを取得
	byteData, err := readModelFile(modelPath)
	if err != nil {
		return nil, err
	}

	// byteデータからモデル情報を作成
	nnModel, err := decodeNNModel(byteData)
	if err != nil {
		return nil, err
	}

	// モデル情報からレイヤー情報を復元する
	return convertNNLayers(nnModel)
}

func convertNNModel(nnLayers *neuralNetwork.NeuralNetworkLayers) (*NNModel, error) {
	nnModel := NewNNModel()

	// レイヤー情報を取得
	for _, layer := range nnLayers.GetLayers() {
		nnData := NewNNData()

		switch convertLayer := layer.(type) {
		case *neuralNetwork.Affine:
			nnData = convertNNDataFromAffine(convertLayer)
		case *neuralNetwork.Tanh:
			nnData.Type = TanhType
		case *neuralNetwork.Relu:
			nnData.Type = ReluType
		case *neuralNetwork.Sigmoid:
			nnData.Type = SigmoidType
		default:
			return nil, errors.New("意図しないレイヤータイプが指定されています.")
		}

		nnModel.Layers = append(nnModel.Layers, nnData)
	}

	// 最終のレイヤーを設定
	nnData := NewNNData()
	nnData.Type = SoftmaxWithLossType
	nnModel.Layers = append(nnModel.Layers, nnData)

	// Optimizerを設定
	nnData = NewNNData()
	optimizer := nnLayers.GetOptimizer()
	switch optimizer.(type) {
	case *neuralNetwork.SGD:
		nnData.Type = SgdType
	default:
		return nil, errors.New("意図しないoptimizerが指定されています.")
	}
	nnModel.Layers = append(nnModel.Layers, nnData)

	return nnModel, nil
}

func convertNNLayers(model *NNModel) (*neuralNetwork.NeuralNetworkLayers, error) {
	nnLayers := neuralNetwork.NewDefaultNeuralNetworkLayers()

	for _, nnData := range model.Layers {
		switch nnData.Type {
		case SgdType:
			// TODO : SGDのパラメーターを設定できるように対応
			nnLayers.SetOptimizer(neuralNetwork.NewSGD())
		case SoftmaxWithLossType:
			nnLayers.SetLastActivationLayer(neuralNetwork.NewSoftmaxWithLoss())
		case SigmoidType:
			nnLayers.Add(neuralNetwork.NewSigmoid())
		case ReluType:
			nnLayers.Add(neuralNetwork.NewRelu())
		case TanhType:
			nnLayers.Add(neuralNetwork.NewTanh())
		case AffineType:
			affine := convertAffineFromNNData(nnData)
			nnLayers.Add(affine)
		default:
			return nil, errors.New("意図しないレイヤータイプが保存されています")
		}
	}

	return nnLayers, nil
}

func convertNNDataFromAffine(affine *neuralNetwork.Affine) NNData {
	nnData := NewNNData()
	nnData.Type = AffineType
	params := affine.GetParams()

	// wの設定
	w := params["w"]
	r, c := w.Dims()
	nnData.Parameter["w"] = NNRawData{r, c, mat.DenseCopyOf(w).RawMatrix().Data}

	// bの設定
	b := params["b"]
	r, c = b.Dims()
	nnData.Parameter["b"] = NNRawData{r, c, mat.DenseCopyOf(b).RawMatrix().Data}

	return nnData
}

func convertAffineFromNNData(data NNData) *neuralNetwork.Affine {
	weight := data.Parameter["w"]
	affine := neuralNetwork.NewAffine(weight.Row, weight.Col)

	// wの設定
	params := make(map[string]mat.Matrix)
	params["w"] = mat.NewDense(weight.Row, weight.Col, weight.RawData)

	// bの設定
	bias := data.Parameter["b"]
	params["b"] = mat.NewDense(bias.Row, bias.Col, bias.RawData)

	affine.UpdateParams(params)
	return affine
}

func encodeNNModel(model *NNModel) ([]byte, error) {
	buf := bytes.NewBuffer(nil)
	if err := gob.NewEncoder(buf).Encode(model); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func decodeNNModel(byteData []byte) (*NNModel, error) {
	model := NewNNModel()
	buf := bytes.NewBuffer(byteData)
	if err := gob.NewDecoder(buf).Decode(model); err != nil {
		return nil, err
	}
	return model, nil
}

func writeModelFile(modelPath string, data []byte) error {
	return ioutil.WriteFile(modelPath, data, 0644)
}

func readModelFile(modelPath string) ([]byte, error) {
	return ioutil.ReadFile(modelPath)
}
