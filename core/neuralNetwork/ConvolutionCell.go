package neuralNetwork

import "gonum.org/v1/gonum/mat"

type Convolution struct {
	w  mat.Matrix
	b  mat.Vector
	x  mat.Matrix
	dw mat.Matrix
	db mat.Vector

	//
}

// NewConvolution :
func NewConvolution() *Convolution {
	return nil
}

func (con *Convolution) Forward(x mat.Matrix) mat.Matrix {
	return nil
}

func (con *Convolution) Backward(dout mat.Matrix) mat.Matrix {
	return nil
}

func (con *Convolution) GetParams() map[string]mat.Matrix {
	return nil
}

func (con *Convolution) GetGradients() map[string]mat.Matrix {
	return nil
}

func (con *Convolution) UpdateParams(params map[string]mat.Matrix) {

}

type MaxPooling struct {
}
