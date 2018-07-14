package neuralNetwork

import (
	"github.com/goMLLibrary/core/image"
	"github.com/goMLLibrary/core/util"
	. "github.com/smartystreets/goconvey/convey"
	"gonum.org/v1/gonum/mat"
	"reflect"
	"testing"
)

func TestConvolution_Forward(t *testing.T) {
	Convey("Given : Convolution層を作成", t, func() {
		Convey("AND : 入力の形（幅×高さ×チャネル)を3*3*2とし、バッチ数を2とする", nil)
		width := 3
		height := 3
		channel := 2
		batch := 2
		inputShape := image.NewNeuralImageShape(width, height, channel, batch)

		Convey("AND : フィルタの形（幅×高さ×チャネル)を2*2*3とする", nil)
		filterShape := image.NewNeuralImageShape(2, 2, 3, 1)

		Convey("AND : paddingが0, strideを1で畳み込み層を作成する", nil)
		stride := 1
		padding := 0
		con := NewConvolution(stride, padding, inputShape, filterShape)

		Convey("AND : 畳み込み層の重み（フィルタ）を1CH目を各値1、2CH目を各値2、3CH目を各値-1とする", nil)
		r := filterShape.Width * filterShape.Height * inputShape.Channel
		c := filterShape.Channel
		w := mat.NewDense(r, c, []float64{
			// 1CH // 2CH // 3CH
			1, 2, -1,
			1, 2, -1,
			1, 2, -1,
			1, 2, -1,
			-1, 1, 2,
			-1, 1, 2,
			-1, 1, 2,
			-1, 1, 2,
		})
		Convey("Then : フィルタの行と列数が想定通りであること", func() {
			actualRowSize, actualColSize := con.w.Dims()
			So(actualRowSize, ShouldEqual, r)
			So(actualColSize, ShouldEqual, c)
		})
		con.w = w

		Convey("AND : バイアスは0とする", nil)
		con.b = mat.NewVecDense(filterShape.Channel, nil)

		Convey("AND : 入力する行列を作成する", nil)
		r = batch
		c = width * height * channel
		input := mat.NewDense(r, c, util.CreateFloatArrayByStep(width*height*channel*batch, 1.0, 1.0))

		Convey("When : 畳み込み層に入力用の行列に対し、Forward処理を実施する", func() {
			out := con.Forward(input)

			Convey("Then : 出力された行列のサイズ(2*12)が意図通りであること", func() {
				r, c := out.Dims()
				expectedRow := batch
				expectedOH := (height+2*padding-filterShape.Height)/stride + 1
				expectedOW := (width+2*padding-filterShape.Width)/stride + 1
				expectedCol := expectedOW * expectedOH * filterShape.Channel
				So(expectedRow, ShouldEqual, r)
				So(expectedCol, ShouldEqual, c)

				Convey("AND : 演算結果の行列の中身が正しいこと", nil)
				values := make([]float64, 0, r*c)
				base1ch1 := []float64{12, 16, 24, 28}
				base1ch2 := []float64{48, 52, 60, 64}
				base2ch1 := []float64{84, 88, 96, 100}
				base2ch2 := []float64{120, 124, 132, 136}

				// 1枚目 1filter(1, 1, 1, 1, -1, -1, -1, -1)
				// 1枚目 2filter(2, 2, 2, 2, 1, 1, 1, 1)
				// 1枚目 3filter(-1, -1, -1, -1, 2, 2, 2, 2)
				values = append(values, util.AddArray(base1ch1, util.ScaleArray(base1ch2, -1))...)
				values = append(values, util.AddArray(util.ScaleArray(base1ch1, 2), base1ch2)...)
				values = append(values, util.AddArray(util.ScaleArray(base1ch1, -1), util.ScaleArray(base1ch2, 2))...)

				// 2枚目 1filter(1, 1, 1, 1, -1, -1, -1, -1)
				// 2枚目 2filter(2, 2, 2, 2, 1, 1, 1, 1)
				// 2枚目 3filter(-1, -1, -1, -1, 2, 2, 2, 2)
				values = append(values, util.AddArray(base2ch1, util.ScaleArray(base2ch2, -1))...)
				values = append(values, util.AddArray(util.ScaleArray(base2ch1, 2), base2ch2)...)
				values = append(values, util.AddArray(util.ScaleArray(base2ch1, -1), util.ScaleArray(base2ch2, 2))...)

				expectedOut := mat.NewDense(expectedRow, expectedCol, values)
				So(reflect.DeepEqual(out, expectedOut), ShouldBeTrue)
			})
		})
	})
}

func TestMaxPooling_Forward(t *testing.T) {
	Convey("Given : MaxPooling層を作成", t, func() {
		Convey("AND : 入力の形（幅×高さ×チャネル）を6*6*2とし、バッチ数は2とする", nil)
		width := 6
		height := 6
		channel := 2
		batch := 2
		inputShape := image.NewNeuralImageShape(width, height, channel, batch)

		Convey("AND : 2*2のMaxPooling層を作成", nil)
		pW := 2
		pH := 2
		mp := NewMaxPooling(inputShape, pW, pH)

		Convey("AND : 入力する行列データを作成", nil)
		r := batch
		c := width * height * channel
		input := mat.NewDense(r, c, util.CreateFloatArrayByStep(width*height*channel*batch, 1.0, 0.5))

		Convey("When : 入力値に対して、プーリング層のForward処理を実施", func() {
			out := mp.Forward(input)

			Convey("Then : 出力地が想定通りであること", func() {
				expectedW := width / pW
				expectedH := height / pH
				r, c := out.Dims()
				So(r, ShouldEqual, batch)
				So(c, ShouldEqual, expectedW*expectedH*channel)

				// 行列の値の確認
				expectedOut := mat.NewDense(batch, expectedW*expectedH*channel, []float64{
					// 1行目
					4.5, 5.5, 6.5, 10.5, 11.5, 12.5, 16.5, 17.5, 18.5, 22.5, 23.5, 24.5, 28.5, 29.5, 30.5, 34.5, 35.5, 36.5,
					// 2行目
					40.5, 41.5, 42.5, 46.5, 47.5, 48.5, 52.5, 53.5, 54.5, 58.5, 59.5, 60.5, 64.5, 65.5, 66.5, 70.5, 71.5, 72.5,
				})
				So(mat.Equal(out, expectedOut), ShouldBeTrue)
			})
		})
	})
}
