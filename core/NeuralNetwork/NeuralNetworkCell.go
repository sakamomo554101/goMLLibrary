package NeuralNetwork

import (
	"github.com/goMLLibrary/core/Util"
	"gonum.org/v1/gonum/mat"
)

// NeuralNetworkLayer : ニューラルネットワークの素子に関するIF
type NeuralNetworkLayer interface {
	// Forward : 順方向伝搬の実施
	Forward(x mat.Matrix) mat.Matrix
	// Backward : 逆方向伝搬の実施
	Backward(dout mat.Matrix) mat.Matrix
	// GetParams : 各種パラメーターを取得
	GetParams() map[string]mat.Matrix
	// GetGradients : 各種勾配を取得
	GetGradient() map[string]mat.Matrix
	// UpdateParams : 各種パラメーターを更新
	UpdateParams(map[string]mat.Matrix)
}

type Affine struct {
	w  mat.Matrix
	b  mat.Vector
	x  mat.Matrix
	dw mat.Matrix
	db mat.Vector
}

// NewAffine : アフィン変換の素子を取得
func NewAffine(inputSize, outputSize int) *Affine {
	w := mat.NewDense(inputSize, outputSize, Util.RandomFloatArray(-1, 1, inputSize*outputSize))
	b := mat.NewVecDense(outputSize, Util.RandomFloatArray(-1, 1, outputSize))
	a := Affine{}
	a.w = w
	a.b = b
	return &a
}

func newAffine(w mat.Matrix, b mat.Vector) *Affine {
	a := Affine{w: w, b: b}
	return &a
}

func (aff *Affine) Forward(x mat.Matrix) mat.Matrix {
	aff.x = x
	batchSize, _ := aff.x.Dims()
	_, outputSize := aff.w.Dims()
	d := mat.NewDense(batchSize, outputSize, nil)
	d.Mul(aff.x, aff.w)
	d.Apply(func(i, j int, v float64) float64 {
		return aff.b.AtVec(j) + v
	}, d)
	return d
}

func (aff *Affine) Backward(dout mat.Matrix) mat.Matrix {
	// dxの計算
	r, c := aff.x.Dims()
	dx := mat.NewDense(r, c, nil)
	dx.Mul(dout, aff.w.T())

	// dwの計算
	r, c = aff.w.Dims()
	dw := mat.NewDense(r, c, nil)
	dw.Mul(aff.x.T(), dout)
	aff.dw = dw

	// dbの計算
	r, c = dout.Dims()
	db := mat.NewVecDense(aff.b.Len(), nil)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			tmpVal := db.AtVec(j)
			db.SetVec(j, tmpVal+dout.At(i, j))
		}
	}
	aff.db = db
	return dx
}

func (aff *Affine) GetParams() map[string]mat.Matrix {
	params := make(map[string]mat.Matrix)
	params["w"] = aff.w
	params["b"] = aff.b
	return params
}

func (aff *Affine) GetGradients() map[string]mat.Matrix {
	grads := make(map[string]mat.Matrix)
	grads["dw"] = aff.dw
	grads["db"] = aff.db
	return grads
}

func (aff *Affine) UpdateParams(params map[string]mat.Matrix) {
	// パラメータのアップデート
	aff.w = params["w"]
	aff.b = mat.DenseCopyOf(params["b"]).ColView(0)

	// 勾配のリセット
	aff.dw = nil
	aff.db = nil
}
