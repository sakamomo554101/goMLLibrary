package neuralNetwork

import (
	"math"
	"testing"

	"github.com/goMLLibrary/core/util"
	. "github.com/smartystreets/goconvey/convey"
	"gonum.org/v1/gonum/mat"
)

func TestSigmoid(t *testing.T) {
	Convey("Given : sigmoidレイヤーが一つ与えられた時", t, func() {
		s := NewSigmoid()
		Convey("AND : 行列サイズを2*3とする", nil)
		r := 2
		c := 3
		Convey("When : 入力行列xが与えられた時", func() {
			x := mat.NewDense(r, c, util.CreateFloatArrayByStep(r*c, 1, 1))
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
			dout := mat.NewDense(r, c, util.CreateFloatArrayByStep(r*c, 0.5, 0.5))
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
			x := mat.NewDense(r, c, util.CreateFloatArrayByStep(r*c, 1, 1))
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
			dout := mat.NewDense(r, c, util.CreateFloatArrayByStep(r*c, 0.5, 0.5))
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
			x := mat.NewDense(r, c, util.CreateFloatArrayByStep(r*c, 1, 1))
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
			dout := mat.NewDense(r, c, util.CreateFloatArrayByStep(r*c, 0.5, 0.5))
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

func TestSoftmaxCrossEntropy(t *testing.T) {
	Convey("Given : SoftmaxCrossEntropyレイヤーが一つ与えられた時", t, func() {
		sce := NewSoftmaxWithLoss()
		Convey("AND : 3次元ベクトルを2個入力するとする（2*3の行列）", nil)
		r := 2 // データ数（バッチ数）
		c := 3 // データの次元
		Convey("When : 入力行列xと正解データが与えられた時", func() {
			// [1,2,3]
			// [4,5,7]
			x := mat.NewDense(r, c, []float64{1, 2, 3, 4, 5, 7})
			// [0,0,1]
			// [1,0,0]
			t := mat.NewDense(r, c, []float64{0, 0, 1, 1, 0, 0})
			loss, _ := sce.Forward(x, t)
			Convey("Then : Forward処理を実施", func() {
				loss_expected := softmaxCrossEntropy_forward(x, t)
				So(loss, ShouldEqual, loss_expected)
			})
			Convey("Then : Backward処理を実施", func() {
				douts := sce.Backward()
				softmax_values := softmax_batch(x)
				douts_expected := softmaxCrossEntropy_backward(softmax_values, t)
				So(mat.Equal(douts, douts_expected), ShouldBeTrue)
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

func softmaxCrossEntropy_forward_1batch(x []float64, tValues []float64) (loss float64) {
	results := softmax(x)
	loss = crossEntropy(results, tValues)
	return
}

func softmaxCrossEntropy_forward(x mat.Matrix, t mat.Matrix) (loss float64) {
	r, _ := x.Dims()
	batch_size := r
	x_dense := mat.DenseCopyOf(x)
	t_dense := mat.DenseCopyOf(t)
	for i := 0; i < r; i++ {
		loss += softmaxCrossEntropy_forward_1batch(x_dense.RawRowView(i), t_dense.RawRowView(i))
	}
	return loss / float64(batch_size)
}

func softmaxCrossEntropy_backward_1batch(outs []float64, tValues []float64, batchSize int) []float64 {
	if len(outs) != len(tValues) {
		panic("softmatCrossEntropy_backward_1batch argument is not match!")
	}
	dx := make([]float64, 0, len(outs))
	for i, out := range outs {
		dxi := (out - tValues[i]) / float64(batchSize)
		dx = append(dx, dxi)
	}
	return dx
}

func softmaxCrossEntropy_backward(out mat.Matrix, t mat.Matrix) mat.Matrix {
	rout, cout := out.Dims()
	rt, ct := t.Dims()
	if rout != rt || cout != ct {
		panic("softmatCrossEntropy_backward argument is not match!")
	}

	dense := mat.NewDense(rout, cout, nil)
	for i := 0; i < rout; i++ {
		vs := softmaxCrossEntropy_backward_1batch(mat.DenseCopyOf(out).RawRowView(i), mat.DenseCopyOf(t).RawRowView(i), rout)
		dense.SetRow(i, vs)
	}
	return dense
}

func crossEntropy(values []float64, tValues []float64) float64 {
	sum := 0.0
	for i, v := range values {
		sum += tValues[i] * math.Log(v+delta)
	}
	return -sum
}

func softmax(values []float64) []float64 {
	// 最大値と合計値(Exp)を求める
	max := values[0]
	sum := 0.0
	for _, v := range values {
		max = math.Max(max, v)
	}
	for _, v := range values {
		sum += math.Exp(v - max)
	}

	// 各列の値を求める
	results := make([]float64, 0, len(values))
	for _, v := range values {
		result := math.Exp(v-max) / sum
		results = append(results, result)
	}
	return results
}

func softmax_batch(values mat.Matrix) mat.Matrix {
	r, c := values.Dims()
	dense := mat.NewDense(r, c, nil)
	for i := 0; i < r; i++ {
		vs := softmax(mat.DenseCopyOf(values).RawRowView(i))
		dense.SetRow(i, vs)
	}
	return dense
}

func max(values []float64) float64 {
	max := values[0]
	for _, v := range values {
		max = math.Max(max, v)
	}
	return max
}
