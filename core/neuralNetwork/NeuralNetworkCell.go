package neuralNetwork

import (
	"gonum.org/v1/gonum/mat"
)

// NeuralNetworkBaseLayer : ニューラルネットワークの素子に関する基本的なIF
type NeuralNetworkBaseLayer interface {
	// Forward : 順方向伝搬の実施
	Forward(x mat.Matrix) mat.Matrix
	// Backward : 逆方向伝搬の実施
	Backward(dout mat.Matrix) mat.Matrix
}

// NeuralNetworkLayer : ニューラルネットワークの素子に関するIF
type NeuralNetworkLayer interface {
	NeuralNetworkBaseLayer
	// GetParams : 各種パラメーターを取得
	GetParams() map[string]mat.Matrix
	// GetGradients : 各種勾配を取得
	GetGradients() map[string]mat.Matrix
	// UpdateParams : 各種パラメーターを更新
	UpdateParams(map[string]mat.Matrix)
}
