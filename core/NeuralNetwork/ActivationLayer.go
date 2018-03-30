package NeuralNetwork

import (
	"gonum.org/v1/gonum/mat"
	"math"
)

// Sigmoid : シグモイド関数
type Sigmoid struct {
	out mat.Matrix
}

// NewSigmoid : シグモイド関数の素子を取得
func NewSigmoid() *Sigmoid {
	sigmoid := Sigmoid{}
	return &sigmoid
}

func (sigmoid *Sigmoid) Forward(x mat.Matrix) mat.Matrix {
	r, c := x.Dims()
	dense := mat.NewDense(r, c, nil) // zero dense
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			val := 1.0 / (1.0 + math.Exp(x.At(i, j)))
			dense.Set(i, j, val)
		}
	}
	sigmoid.out = dense
	return dense
}

func (sigmoid *Sigmoid) Backward(dout mat.Matrix) mat.Matrix {
	r, c := dout.Dims()
	dense := mat.NewDense(r, c, nil) // zero dense
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			val := dout.At(i, j) * (1 - sigmoid.out.At(i, j)) * sigmoid.out.At(i, j)
			dense.Set(i, j, val)
		}
	}
	return dense
}
