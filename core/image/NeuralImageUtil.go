package image

import (
	"errors"
	"gonum.org/v1/gonum/mat"
)

// Im2col : 各画像データ（複数チャネルを保持しているため、3次元データ）をベクトルデータに変換する
// 出力値は各画像データ分のベクトルとなるため、行列データとなる
// input : 複数チャネルを持つ画像を複数入力
// filterH : フィルタの高さ
// filterW : フィルタの幅
// stride : ストライドのサイズ
// padding : パディングのサイズ
func Im2col(input ImagesWithChannel, filterH int, filterW int, stride int, padding int) mat.Matrix {
	// 2次元の行列サイズを確定させる
	ow := (input.GetWidth()+2*padding-filterW)/stride + 1
	oh := (input.GetHeight()+2*padding-filterH)/stride + 1
	batch := input.GetBatchCount()
	channel := input.GetChannel()
	matrixW := filterW * filterH * channel
	matrixH := ow * oh * batch

	// 2次元の行列を初期化
	dense := mat.NewDense(matrixW, matrixH, nil)

	// 4次元データを2次元情報に変換する

	return nil
}

// getOutSize : フィルタをかけた際の出力サイズを計算する
// サイズを計算した際に割り切れなかった場合はエラーを返す
func getOutSize(inputSize int, filterSize int, stride int, padding int) (int, error) {
	n := inputSize + 2*padding - filterSize
	if n < 0 {
		return 0, errors.New("getOutSize : 分子の計算結果が負になっているため、入力値が誤っています")
	}
	if n%stride != 0 {
		return 0, errors.New("getOutSize : 出力サイズが小数となっているため、入力値が誤っています")
	}
	return n/stride + 1, nil
}
