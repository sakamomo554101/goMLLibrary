package util

import "gonum.org/v1/gonum/mat"

// Transpose : 転置行列の作成
// TODO : Matrixをdenseに変換して、Tを実行すれば正常に転置されるかも(よってこのAPIは不要)
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

// Reshape : 行列を指定した形式の行列に変換する
// ex) 6 * 6行列 => 2 * 19行列
func Reshape(base mat.Matrix, r, c int) mat.Matrix {
	bR, bC := base.Dims()
	if bR*bC != r*c {
		panic("Reshape error : 変換前と変換後の要素数に差分があります")
	}

	rawValues := RawValues(base)
	return mat.NewDense(r, c, rawValues)
}

// RawValues : matrixの元データをfloat64の配列に格納
// 配列サイズはrow * colとなる
func RawValues(a mat.Matrix) []float64 {
	dense := mat.DenseCopyOf(a)
	r, c := dense.Dims()
	rawValues := make([]float64, 0, r*c)

	for i := 0; i < r; i++ {
		rawValues = append(rawValues, dense.RawRowView(i)...)
	}
	return rawValues
}
