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
	out mat.Matrix
}

// NewRelu : Relu関数の素子を取得
func NewRelu() *Relu {
	r := &Relu{}
	return r
}

func (relu *Relu) Forward(x mat.Matrix) mat.Matrix {
	r, c := x.Dims()
	dense := mat.NewDense(r, c, nil) // zero matrix
	for i := 0; i < r ; i++ {
		for j := 0; j < c; j++ {
			if x.At(i, j) > 0 {
				dense.At(i, j) = x.At(i, j)
			}
		}
	}
	relu.out = dense
	return dense
}

func (relu *Relu) Backward(dout mat.Matrix) mat.Matrix {
	r, c := dout.Dims()
	dense := mat.NewDense(r, c, nil)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			if relu.out.At(i, j) > 0 {
				dense.At(i, j) = dout.At(i, j)
			}
		}
	}
	return dense
}

// Tanh : Tanh関数
type Tanh struct {
	out mat.Matrix
}

// NewTanh : Tanh関数の素子を取得
func NewTanh() *Tanh {
	t := &Tanh{}
	return t
}

func (tanh *Tanh) Forward(x mat.Matrix) mat.Matrix {
	r, c := x.Dims()
	dense := mat.NewDense(r, c, nil)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			dense.At(i,j) = math.Tanh(x.At(i,j))
		}
	}
	tanh.out = dense
	return dense
}

func (tanh *Tanh) Backward(dout mat.Matrix) mat.Matrix {
	r, c := dout.Dims()
	dense := mat.NewDense(r, c, nil)
	for i := 0; i < c; i++ {
		for j := 0; j < r; j++ {
			dense.At(i,j) = dout.At(i,j) * (1- math.Pow(tanh.out.At(i,j), 2))
		}
	}
	return dense
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

