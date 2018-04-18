package util

import "gonum.org/v1/gonum/mat"

// Transpose : 転置行列の作成
func Transpose(a mat.Matrix) mat.Matrix {
	r, c := a.Dims()
	dense := mat.NewDense(c, r, nil)
	for i := 0; i < c; i++ {
		for j := 0; j < r; j++ {
			dense.Set(i, j, a.At(j, i))
		}
	}
	return dense
}
