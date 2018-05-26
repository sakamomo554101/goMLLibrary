package image

import "gonum.org/v1/gonum/mat"

// Image : ニューラルネットワークでの画像データ（1チャンネル分）を格納する配列データ
type Image [][]float64

// NewImage : 単一チャネルの画像データを格納する配列データを作成
// パディング対応しており、パディング分は0でデータを埋める
// input : 画像の元データを格納した配列データ
// w : 幅
// h : 高さ
// padding : パディングサイズ
func NewImage(input []float64, w int, h int, padding int) Image {
	if len(input) != w*h {
		panic("入力された画像データと指定した幅・高さがマッチしてません")
	}
	if padding < 0 {
		panic("パディングサイズに負の値が設定されています")
	}
	image := make([][]float64, h+padding*2, h+padding*2)
	for i := 0; i < len(image); i++ {
		row := make([]float64, 0, w+padding*2)
		if i >= padding && i < len(image)-padding {
			// パディング分は0に設定するために複数回appendしている
			// TODO : もっとスマートな書き方がありそう
			row = append(row, make([]float64, padding, padding)...)
			row = append(row, input[(i-padding)*w:(i-padding+1)*w]...)
			row = append(row, make([]float64, padding, padding)...)
		} else {
			// パディング箇所のため、0でうめる
			row = append(row, make([]float64, w+padding*2, w+padding*2)...)
		}
		image[i] = row
	}
	return image
}

// GetWidth : 画像の幅を取得
func (img Image) GetWidth() int {
	return len(img[0])
}

// GetHeight : 画像の高さを取得
func (img Image) GetHeight() int {
	return len(img)
}

// getIm2ColWindow : im2colで計算するために画像をカーネルサイズ×出力面積とする
func (img Image) getIm2ColWindow(ow int, oh int, stride int, karnelSize int) [][]float64 {
	window := make([][]float64, 0, ow*oh)
	rowSize := karnelSize * karnelSize
	for h := 0; h < oh; h += stride {
		for w := 0; w < ow; w += stride {
			row := make([]float64, 0, rowSize)
			for k := 0; k < karnelSize; k++ {
				row = append(row, img[h][w:w+karnelSize+1]...)
			}
			window = append(window, row)
		}
	}
	return window
}

// ImageWithChannel : 複数チャネル（RGBなど）を持つ画像データを格納する配列データ
type ImageWithChannel []Image

// NewImageWithChannel : 複数チャネルを持つ画像データを作成
// パディング対応しており、パディング分は0でデータを埋める
// input : 画像の元データを格納した配列データ
// w : 幅
// h : 高さ
// c : チャネル数
// padding : パディングサイズ
func NewImageWithChannel(input []float64, w int, h int, c int, padding int) ImageWithChannel {
	if len(input) != w*h*c {
		panic("入力された画像データと指定した幅・高さ・チャネル数がマッチしてません")
	}
	iwc := make([]Image, 0, c)

	for i := 0; i < c; i++ {
		image := NewImage(input[i*w*h:(i+1)*w*h], w, h, padding)
		iwc = append(iwc, image)
	}
	return iwc
}

// GetWidth : 画像の幅を取得
func (iwc ImageWithChannel) GetWidth() int {
	return iwc[0].GetWidth()
}

// GetHeight : 画像の高さを取得
func (iwc ImageWithChannel) GetHeight() int {
	return iwc[0].GetHeight()
}

// GetChennel : 画像のチャネル数を取得
func (iwc ImageWithChannel) GetChennel() int {
	return len(iwc)
}

// im2Col : 3次元情報を（フィルタでの計算を行うために）2次元に変換
// 行サイズ：出力幅×出力高さ
// 列サイズ：フィルタサイズ×フィルタサイズ×チャネル数
func (iwc ImageWithChannel) im2Col(ow int, oh int, stride int, karnelSize int) ([][]float64, error) {
	col := make([][]float64, 0, ow*oh)

	// チャネル毎のデータを結合する
	// 行：ow*ohのサイズ
	// 列：filterSize*filterSize*channelのサイズ
	for i := 0; i < ow*oh; i++ {
		row := make([]float64, karnelSize*karnelSize*iwc.GetChennel())
		for c := 0; c < iwc.GetChennel(); c++ {
			row = append(row, iwc[c].getIm2ColWindow(ow, oh, stride, karnelSize)[i]...)
		}
		col = append(col, row)
	}
	return col, nil
}

// ImagesWithChannel : 複数チャネルを持つ画像データを複数格納した配列データ
type ImagesWithChannel []ImageWithChannel

// NewImagesWithChannel : 複数チャネルを持つ画像データを作成
// パディング対応しており、パディング分は0でデータを埋める
// input : 画像の元データを格納した配列データ
// w : 幅
// h : 高さ
// c : チャネル数
// batch : 画像データ数（バッチ数）
// padding : パディングサイズ
func NewImagesWithChannel(input []float64, w int, h int, c int, batch int, padding int) ImagesWithChannel {
	if len(input) != w*h*c*batch {
		panic("入力された画像データと指定した幅・高さ・チャネル数・画像数がマッチしてません")
	}
	iwcb := make([]ImageWithChannel, 0, batch)

	for i := 0; i < batch; i++ {
		imageWithChannel := NewImageWithChannel(input[i*w*h*c:(i+1)*w*h*c], w, h, c, padding)
		iwcb = append(iwcb, imageWithChannel)
	}
	return iwcb
}

// GetWidth : 画像の幅を取得
func (iwcb ImagesWithChannel) GetWidth() int {
	return iwcb[0].GetWidth()
}

// GetHeight : 画像の高さを取得
func (iwcb ImagesWithChannel) GetHeight() int {
	return iwcb[0].GetHeight()
}

// GetChennel : 画像のチャネル数を取得
func (iwcb ImagesWithChannel) GetChannel() int {
	return iwcb[0].GetChennel()
}

// GetBatchCount : 画像の数を取得
func (iwcb ImagesWithChannel) GetBatchCount() int {
	return len(iwcb)
}

// Im2Col : 4次元情報を（フィルタ計算のために）2次元情報へ変換する
// 行サイズ：出力幅×出力高さ×データ数
// 列サイズ：フィルタサイズ×フィルタサイズ×チャネル数
func (iwcb ImagesWithChannel) Im2Col(stride int, karnelSize int) (mat.Matrix, error) {
	ow, err := getOutSize(iwcb.GetWidth(), karnelSize, stride, 0)
	if err != nil {
		return nil, err
	}
	oh, err := getOutSize(iwcb.GetHeight(), karnelSize, stride, 0)
	if err != nil {
		return nil, err
	}

	rowSize := ow * oh * iwcb.GetBatchCount()
	colSize := karnelSize * karnelSize * iwcb.GetChannel()
	dense := mat.NewDense(rowSize, colSize, nil)

	// 各画像毎にim2colを実施する

	return dense, nil
}
