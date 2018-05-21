package image

import "gonum.org/v1/gonum/mat"

// Im2col : 各画像データ（複数チャネルを保持しているため、3次元データ）をベクトルデータに変換する
// 出力値は各画像データ分のベクトルとなるため、行列データとなる
// input : 複数チャネルを持つ画像を複数入力
// filterH : フィルタの高さ
// filterW : フィルタの幅
// stride : ストライドのサイズ
// padding : パディングのサイズ
func Im2col(input ImagesWithChannel, filterH int, filterW int, stride int, padding int) mat.Matrix {
	// 2次元の行列サイズを確定させる
	/*ow := (input.GetWidth()+2*padding-filterW)/stride + 1
	oh := (input.GetHeight()+2*padding-filterH)/stride + 1
	batch := input.GetBatchCount()
	channel := input.GetChannel()
	matrixW := filterW * filterH * channel
	matrixH := ow * oh * batch

	// 2次元の行列を初期化
	dense := mat.NewDense(matrixW, matrixH, nil)*/
	return nil
}
