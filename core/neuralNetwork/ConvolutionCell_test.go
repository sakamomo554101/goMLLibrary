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
			})
		})
	})
}
