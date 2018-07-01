package neuralNetwork

import (
	"github.com/goMLLibrary/core/image"
	"github.com/goMLLibrary/core/util"
	"gonum.org/v1/gonum/mat"
)

type Convolution struct {
	// parameter
	w mat.Matrix
	b mat.Vector

	// input and diff
	x  mat.Matrix
	dw mat.Matrix
	db mat.Vector

	// filter setting
	stride      int
	padding     int
	filterShape image.NeuralImageShape

	// input parameter
	inputShape image.NeuralImageShape
}

// NewConvolution : Convolution Layerを作成
func NewConvolution(stride, padding int, inputShape image.NeuralImageShape, filterShape image.NeuralImageShape) *Convolution {
	con := Convolution{stride: stride, padding: padding, filterShape: filterShape, inputShape: inputShape}

	// 重みの行列サイズ（2次元）は以下の計算式
	// 重み行数：フィルタ幅×フィルタ高×入力のチャネル数
	// 重み列数：フィルタ数（フィルタのチャネル数）
	wRow := filterShape.Width * filterShape.Height * inputShape.Channel
	wCol := filterShape.Channel
	con.w = mat.NewDense(wRow, wCol, util.NormRandomArray(0.01, wRow*wCol))
	con.b = mat.NewVecDense(wCol, util.NormRandomArray(0.01, wCol))
	return &con
}

func (con *Convolution) Forward(x mat.Matrix) mat.Matrix {
	// 入力データは画像データ×画像数のため、まず画像の構造体に変換する
	iwcb := image.NewImagesWithChannelFromMatrix(x, con.inputShape.Width, con.inputShape.Height, con.inputShape.Channel, con.padding)

	// 画像を4次元からフィルタ演算をしやすいように2次元に変換
	img, err := image.Im2col(iwcb, con.filterShape.Width, con.stride)
	if err != nil {
		panic("Convolution#Forward : forward処理に失敗しました. err = " + err.Error())
	}

	// 画像データとフィルタデータの内積を計算する
	r, _ := img.Dims()
	_, c := con.w.Dims()
	dense := mat.NewDense(r, c, nil)
	dense.Mul(img, con.w)

	// TODO : バイアスを加算する

	// 行列を整形する（row : batch, col : ow * oh * FN）
	ow, err := image.GetOutSize(con.inputShape.Width, con.filterShape.Width, con.stride, con.padding)
	if err != nil {
		panic("Convolution#Forward : 出力幅の取得に失敗しました")
	}
	oh, err := image.GetOutSize(con.inputShape.Height, con.filterShape.Height, con.stride, con.padding)
	if err != nil {
		panic("Convolution#Forward : 出力高の取得に失敗しました")
	}
	return util.Reshape(dense, con.inputShape.BatchSize, ow*oh*con.filterShape.Channel)
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
