package neuralNetwork

import (
	"github.com/goMLLibrary/core/image"
	"github.com/goMLLibrary/core/util"
	"gonum.org/v1/gonum/mat"
)

// Convolution : 畳み込み層の機能を持つモジュール
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
	con.x = x

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

	// バイアスを加算する
	util.AddVecToMatrixCol(dense, con.b)

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
	// dxの計算
	r, c := con.x.Dims()
	dx := mat.NewDense(r, c, nil)
	dx.Mul(dout, util.Transpose(con.w))

	// dwの計算
	r, c = con.w.Dims()
	dw := mat.NewDense(r, c, nil)
	dw.Mul(util.Transpose(con.x), dout)
	con.dw = dw

	// dbの計算
	con.db = util.SumEachCol(dout)
	return dx
}

func (con *Convolution) GetParams() map[string]mat.Matrix {
	params := make(map[string]mat.Matrix)
	params["w"] = con.w
	params["b"] = con.b
	return params
}

func (con *Convolution) GetGradients() map[string]mat.Matrix {
	grads := make(map[string]mat.Matrix)
	grads["w"] = con.dw
	grads["b"] = con.db
	return grads
}

func (con *Convolution) UpdateParams(params map[string]mat.Matrix) {
	// パラメータのアップデート
	con.w = params["w"]
	con.b = mat.DenseCopyOf(params["b"]).ColView(0)

	// 勾配のリセット
	con.dw = nil
	con.db = nil
}

// MaxPooling : 最大値を伝達するプーリング層
type MaxPooling struct {
	// filter setting
	poolingShape image.NeuralImageShape

	// input parameter
	inputShape image.NeuralImageShape

	maxArgs []int
}

// NewMaxPooling : MaxPooling層のインスタンスを取得
func NewMaxPooling(inputShape image.NeuralImageShape, poolingW, poolingH int) *MaxPooling {
	poolingShape := image.NewNeuralImageShape(poolingW, poolingH, 1, 1)
	return &MaxPooling{poolingShape: poolingShape, inputShape: inputShape}
}

func (mp *MaxPooling) Forward(x mat.Matrix) mat.Matrix {
	padding := 0
	stride := mp.poolingShape.Width

	// 入力データは画像データ×画像数のため、まず画像の構造体に変換する
	iwcb := image.NewImagesWithChannelFromMatrix(x, mp.inputShape.Width, mp.inputShape.Height, mp.inputShape.Channel, padding)

	// 画像を4次元からフィルタ演算をしやすいように2次元に変換
	img, err := image.Im2col(iwcb, mp.poolingShape.Width, stride)
	if err != nil {
		panic("Pooling#Forward : forward処理に失敗しました. err = " + err.Error())
	}

	// プーリングのサイズに合わせて、行列をreshapeする
	// 変換前 => row : ow * oh * N , col : poolingW * poolingH * Channel
	// 変換後 => row : ow * oh * Channel * N, col : poolingW * poolingH
	ow, err := image.GetOutSize(mp.inputShape.Width, mp.poolingShape.Width, stride, padding)
	if err != nil {
		panic("Pooling#Forward : 出力幅の取得に失敗しました")
	}
	oh, err := image.GetOutSize(mp.inputShape.Height, mp.poolingShape.Height, stride, padding)
	if err != nil {
		panic("Pooling#Forward : 出力高の取得に失敗しました")
	}
	r := ow * oh * iwcb.GetChannel() * iwcb.GetBatchCount()
	c := mp.poolingShape.Width * mp.poolingShape.Height
	img = util.Reshape(img, r, c)

	// プーリング処理を行う
	vec, maxArgs := util.MaxEachRow(img)
	mp.maxArgs = maxArgs

	// データの順序を直すため、ベクトルを下記の行列に直す
	// 変換前 => ow * oh * Channel * N
	// 変換後 => row : ow * oh * N, col : channel
	img = util.Reshape(vec, ow*oh*iwcb.GetBatchCount(), iwcb.GetChannel())

	// 各列が各チャネルの出力データとなるため、列データを取得する
	dense := mat.DenseCopyOf(img)
	rawData := make([]float64, 0, r*c)
	for index := 0; index < iwcb.GetBatchCount(); index++ {
		// 各列はchannel毎のow * oh * Nの形なため、
		// 各データ分だけ、入れ替える
		for j := 0; j < iwcb.GetChannel(); j++ {
			col := dense.ColView(j)
			vec := mat.VecDenseCopyOf(col)
			rawData = append(rawData, vec.RawVector().Data[index*ow*oh:(index+1)*ow*oh]...)
		}
	}

	// float64の配列を出力する行列に変換する
	// 変換前 => ow * oh * Channel * N
	// 変換後 => row : N, col : ow * oh * Channel
	img = mat.NewDense(iwcb.GetBatchCount(), ow*oh*iwcb.GetChannel(), rawData)
	return img
}

/*func ReshapeFromIm2ColMatrix(img mat.Matrix) mat.Matrix {
	// 各列が各チャネルの出力データとなるため、列データを取得する
	dense := mat.DenseCopyOf(img)
	rawData := make([]float64, 0, r*c)
	for index := 0; index < iwcb.GetBatchCount(); index++ {
		// 各列はchannel毎のow * oh * Nの形なため、
		// 各データ分だけ、入れ替える
		for j := 0; j < iwcb.GetChannel(); j++ {
			col := dense.ColView(j)
			vec := mat.VecDenseCopyOf(col)
			rawData = append(rawData, vec.RawVector().Data[index*ow*oh:(index+1)*ow*oh]...)
		}
	}

	// float64の配列を出力する行列に変換する
	// 変換前 => ow * oh * Channel * N
	// 変換後 => row : N, col : ow * oh * Channel
	return mat.NewDense(iwcb.GetBatchCount(), ow*oh*iwcb.GetChannel(), rawData)
}*/

func (mp *MaxPooling) Backward(dout mat.Matrix) mat.Matrix {
	return nil
}
