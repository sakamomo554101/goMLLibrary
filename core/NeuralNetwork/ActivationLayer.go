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

// Relu : Relu関数
type Relu struct {
}

// NewRelu : Relu関数の素子を取得
func NewRelu() *Relu {
	r := &Relu{}
	return r
}

func (r *Relu) Forward(x mat.Matrix) mat.Matrix {
	return nil
}

func (r *Relu) Backward(dout mat.Matrix) mat.Matrix {
	return nil
}

// Tanh : Tanh関数
type Tanh struct {
}

// NewTanh : Tanh関数の素子を取得
func NewTanh() *Tanh {
	t := &Tanh{}
	return t
}

func (tanh *Tanh) Forward(x mat.Matrix) mat.Matrix {
	return nil
}

func (tanh *Tanh) Backward(dout mat.Matrix) mat.Matrix {
	return nil
}

type Softmax struct {
}

func NewSoftmax() *Softmax {
	s := &Softmax{}
	return s
}

func (s *Softmax) Forward(x mat.Matrix) mat.Matrix {
	return nil
}

func (s *Softmax) Backward(dout mat.Matrix) mat.Matrix {
	return nil
}
