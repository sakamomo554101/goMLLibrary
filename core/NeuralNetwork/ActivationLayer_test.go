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
		Convey("AND : 行列サイズを2*3とする", nil)
		r := 2
		c := 3
		Convey("When : 入力行列xが与えられた時", func() {
			x := mat.NewDense(r, c, createFloatArrayByStep(r*c, 1, 1))
			out := s.Forward(x)
			Convey("Then : Forward処理を実施", func() {
				act_r, act_c := out.Dims()
				So(act_r, ShouldEqual, r)
				So(act_c, ShouldEqual, c)
				for i := 0; i < r; i++ {
					for j := 0; j < c; j++ {
						So(out.At(i, j), ShouldEqual, sigmoid_forward(x.At(i, j)))
					}
				}
			})

			Convey("AND : 誤差doutが与えられた時", nil)
			dout := mat.NewDense(r, c, createFloatArrayByStep(r*c, 0.5, 0.5))
			Convey("Then : Backward処理を実施", func() {
				out := s.Backward(dout)
				act_r, act_c := out.Dims()
				So(act_r, ShouldEqual, r)
				So(act_c, ShouldEqual, c)
				for i := 0; i < r; i++ {
					for j := 0; j < c; j++ {
						So(out.At(i, j), ShouldEqual, sigmoid_backward(x.At(i, j), dout.At(i, j)))
					}
				}
			})
		})
	})
}

func TestRelu(t *testing.T) {
	Convey("Given : Reluレイヤーが一つ与えられた時", t, func() {
		relu := NewRelu()
		Convey("AND : 行列サイズを3*2とする", nil)
		r := 3
		c := 2
		Convey("When : 入力行列xが与えられた時", func() {
			x := mat.NewDense(r, c, createFloatArrayByStep(r*c, 1, 1))
			out := relu.Forward(x)
			Convey("Then : Forward処理を実施", func() {
				act_r, act_c := out.Dims()
				So(act_r, ShouldEqual, r)
				So(act_c, ShouldEqual, c)
				for i := 0; i < r; i++ {
					for j := 0; j < c; j++ {
						So(out.At(i, j), ShouldEqual, relu_forward(x.At(i, j)))
					}
				}
			})

			Convey("AND : 誤差doutが与えられた時", nil)
			dout := mat.NewDense(r, c, createFloatArrayByStep(r*c, 0.5, 0.5))
			Convey("Then : Backward処理を実施", func() {
				out := relu.Backward(dout)
				act_r, act_c := out.Dims()
				So(act_r, ShouldEqual, r)
				So(act_c, ShouldEqual, c)
				for i := 0; i < r; i++ {
					for j := 0; j < c; j++ {
						So(out.At(i, j), ShouldEqual, relu_backward(x.At(i, j), dout.At(i, j)))
					}
				}
			})
		})
	})
}

func TestTanh(t *testing.T) {
	Convey("Given : Tanhレイヤーが一つ与えられた時", t, func() {
		tanh := NewTanh()
		Convey("AND : 行列サイズを3*2とする", nil)
		r := 3
		c := 2
		Convey("When : 入力行列xが与えられた時", func() {
			x := mat.NewDense(r, c, createFloatArrayByStep(r*c, 1, 1))
			out := tanh.Forward(x)
			Convey("Then : Forward処理を実施", func() {
				act_r, act_c := out.Dims()
				So(act_r, ShouldEqual, r)
				So(act_c, ShouldEqual, c)
				for i := 0; i < r; i++ {
					for j := 0; j < c; j++ {
						So(out.At(i, j), ShouldEqual, tanh_forward(x.At(i, j)))
					}
				}
			})

			Convey("AND : 誤差doutが与えられた時", nil)
			dout := mat.NewDense(r, c, createFloatArrayByStep(r*c, 0.5, 0.5))
			Convey("Then : Backward処理を実施", func() {
				out := tanh.Backward(dout)
				act_r, act_c := out.Dims()
				So(act_r, ShouldEqual, r)
				So(act_c, ShouldEqual, c)
				for i := 0; i < r; i++ {
					for j := 0; j < c; j++ {
						checkValue(out.At(i, j), tanh_backward(x.At(i, j), dout.At(i, j)), math.Pow10(-8))
					}
				}
			})
		})
	})
}

func checkValue(act, exp, diff float64) {
	So(math.Abs(act-exp), ShouldBeLessThan, diff)
}

func sigmoid_forward(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func sigmoid_backward(x float64, dout float64) float64 {
	return dout * (1.0 - sigmoid_forward(x)) * sigmoid_forward(x)
}

func relu_forward(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

func relu_backward(x float64, dout float64) float64 {
	if x > 0 {
		return dout
	}
	return 0
}

func tanh_forward(x float64) float64 {
	return math.Tanh(x)
}

func tanh_backward(x float64, dout float64) float64 {
	return dout * 4 / math.Pow(math.Exp(x)+math.Exp(-x), 2)
}
