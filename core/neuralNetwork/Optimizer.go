package neuralNetwork

import (
	"gonum.org/v1/gonum/mat"
)

// Optimizer : パラメーターと勾配情報からパラメーターの最適化を行うIF
type Optimizer interface {
	// Update : パラメーターを勾配情報を元に最適化(更新)する
	Update(params map[string]mat.Matrix, grads map[string]mat.Matrix)
}

// SGD : 確率的勾配降下法を行うOptimizer
type SGD struct {
	lr float64
}

// SGDOption : SGDのオプション
type SGDOption func(*SGD)

const (
	// DefaultLearningRate : デフォルトの学習率の値
	DefaultLearningRate = 0.01
)

// NewSGD : SGDを取得するAPI
// 初期化時にオプション指定が可能
func NewSGD(options ...SGDOption) *SGD {
	sgd := SGD{}
	sgd.lr = DefaultLearningRate

	// オプションが設定されていれば利用
	for _, opt := range options {
		opt(&sgd)
	}
	return &sgd
}

// WithSGDLearningRate : SGDの学習率指定のオプションを取得
func WithSGDLearningRate(lr float64) SGDOption {
	return func(sgd *SGD) {
		sgd.lr = lr
	}
}

func (sgd *SGD) Update(params map[string]mat.Matrix, grads map[string]mat.Matrix) {
	for key, _ := range params {
		//r, c := params[key].Dims()
		dense := mat.DenseCopyOf(params[key])

		// 学習率分だけ勾配を拡縮
		dense.Apply(func(i, j int, v float64) float64 {
			return v * sgd.lr
		}, grads[key])

		// 重みから勾配分（学習率を考慮）だけ差をとる
		dense.Sub(params[key], dense)

		// paramに戻す
		params[key] = dense
	}
}
