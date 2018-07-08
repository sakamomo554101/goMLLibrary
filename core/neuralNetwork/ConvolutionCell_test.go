package neuralNetwork

import (
	"github.com/goMLLibrary/core/image"
	"github.com/goMLLibrary/core/util"
	. "github.com/smartystreets/goconvey/convey"
	"gonum.org/v1/gonum/mat"
	"testing"
)

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
