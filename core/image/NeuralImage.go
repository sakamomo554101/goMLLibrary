package image

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
