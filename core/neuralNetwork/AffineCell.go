package neuralNetwork

import (
	"github.com/goMLLibrary/core/util"
	"gonum.org/v1/gonum/mat"
)

type Affine struct {
	w  mat.Matrix
	b  mat.Vector
	x  mat.Matrix
	dw mat.Matrix
	db mat.Vector
}

// NewAffine : アフィン変換の素子を取得
func NewAffine(inputSize, outputSize int) *Affine {
	w := mat.NewDense(inputSize, outputSize, util.NormRandomArray(0.01, outputSize*inputSize))
	b := mat.NewVecDense(outputSize, util.NormRandomArray(0.01, outputSize))
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
	util.AddVecToMatrixCol(d, aff.b)
	return d
}

func (aff *Affine) Backward(dout mat.Matrix) mat.Matrix {
	// dxの計算
	r, c := aff.x.Dims()
	dx := mat.NewDense(r, c, nil)
	dx.Mul(dout, util.Transpose(aff.w))

	// dwの計算
	r, c = aff.w.Dims()
	dw := mat.NewDense(r, c, nil)
	dw.Mul(util.Transpose(aff.x), dout)
	aff.dw = dw

	// dbの計算
	aff.db = util.SumEachCol(dout)
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
	grads["w"] = aff.dw
	grads["b"] = aff.db
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
