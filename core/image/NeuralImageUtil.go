package image

import (
	"errors"
	"gonum.org/v1/gonum/mat"
)

// Im2col : 各画像データ（複数チャネルを保持しているため、3次元データ）をベクトルデータに変換する
// 出力値は各画像データ分のベクトルとなるため、行列データとなる
// input : 複数チャネルを持つ画像を複数入力
// filterSize : フィルタサイズ（高さと幅、ともに同じ大きさ）
// stride : ストライドのサイズ
func Im2col(input ImagesWithChannel, filterSize int, stride int) (mat.Matrix, error) {
	ow, err := GetOutSize(input.GetWidth(), filterSize, stride, 0)
	if err != nil {
		return nil, err
	}
	oh, err := GetOutSize(input.GetHeight(), filterSize, stride, 0)
	if err != nil {
		return nil, err
	}

	rowSize := ow * oh * input.GetBatchCount()
	colSize := filterSize * filterSize * input.GetChannel()
	dense := mat.NewDense(rowSize, colSize, nil)

	// 各画像毎にim2colを実施する
	// 取得した2次元データを順番にmatrixに追加する
	for b, iwc := range input {
		cols := iwc.im2Col(ow, oh, stride, filterSize)
		for i, col := range cols {
			dense.SetRow(b*ow*oh+i, col)
		}
	}
	return dense, nil
}

/*func Col2im(col mat.Matrix, inputSize []int, filterSize int, stride int) (ImagesWithChannel, error) {

}*/

// GetOutSize : フィルタをかけた際の出力サイズを計算する
// サイズを計算した際に割り切れなかった場合はエラーを返す
func GetOutSize(inputSize int, filterSize int, stride int, padding int) (int, error) {
	n := inputSize + 2*padding - filterSize
	if n < 0 {
		return 0, errors.New("GetOutSize : 分子の計算結果が負になっているため、入力値が誤っています")
	}
	if n%stride != 0 {
		return 0, errors.New("GetOutSize : 出力サイズが小数となっているため、入力値が誤っています")
	}
	return n/stride + 1, nil
}
