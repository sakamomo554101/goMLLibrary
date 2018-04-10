package NeuralNetwork

import (
	. "github.com/smartystreets/goconvey/convey"
	"gonum.org/v1/gonum/mat"
	"math"
	"testing"
)

func TestSigmoid(t *testing.T) {
	Convey("Given : sigmoidレイヤーが一つ与えられた時", t, func() {
		s := NewSigmoid()
		Convey("When : 入力行列x(2*3)が与えられた時", func() {
			x := mat.NewDense(2, 3, createFloatArrayByStep(6, 1, 1))
			out := s.Forward(x)
			Convey("Then : Forward処理を実施", func() {
				r, c := out.Dims()
				So(r, ShouldEqual, 2)
				So(c, ShouldEqual, 3)
				for i := 0; i < r; i++ {
					for j := 0; j < c; j++ {
						So(out.At(i, j), ShouldEqual, sigmoid_forward(x.At(i, j)))
					}
				}
			})

			Convey("AND : 誤差dout(2*3)が与えられた時", nil)
			dout := mat.NewDense(2, 3, createFloatArrayByStep(6, 0.5, 0.5))
			Convey("Then : Backward処理を実施", func() {
				out := s.Backward(dout)
				r, c := out.Dims()
				So(r, ShouldEqual, 2)
				So(c, ShouldEqual, 3)
				for i := 0; i < r; i++ {
					for j := 0; j < c; j++ {
						So(out.At(i, j), ShouldEqual, sigmoid_backward(x.At(i, j), dout.At(i, j)))
					}
				}
			})
		})
	})
}

func sigmoid_forward(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func sigmoid_backward(x float64, dout float64) float64 {
	return dout * (1.0 - sigmoid_forward(x)) * sigmoid_forward(x)
}
