package neuralNetwork

import (
	"testing"

	"github.com/goMLLibrary/core/util"
	. "github.com/smartystreets/goconvey/convey"
	"gonum.org/v1/gonum/mat"
)

func TestSGD(t *testing.T) {
	Convey("Given : 1つの重み、1つのバイアスが与えられた時", t, func() {
		params := make(map[string]mat.Matrix)
		grads := make(map[string]mat.Matrix)
		Convey("AND 重み行列は4*3行列で各値は1-12とし、勾配は0-5.5(0.5刻み）とする", nil)
		w := mat.NewDense(4, 3, util.CreateFloatArrayByStep(12, 1.0, 1.0))
		dw := mat.NewDense(4, 3, util.CreateFloatArrayByStep(12, 0, 0.5))
		params["w"] = w
		grads["w"] = dw
		Convey("AND バイアスは3次元で各値は0-2とし、勾配は-2-0とする", nil)
		b := mat.NewVecDense(3, util.CreateFloatArrayByStep(3, 0, 1.0))
		db := mat.NewVecDense(3, util.CreateFloatArrayByStep(3, -2, 1.0))
		params["b"] = b
		grads["b"] = db
		Convey("When : SGDの学習率0.1で初期化", func() {
			sgd := NewSGD(WithSGDLearningRate(0.1))
			Convey("Then : Optimizerでupdateを実施", func() {
				sgd.Update(params, grads)
				// param["w"]
				// [1, 1.95, 2.9]
				// [3.85, 4.8, 5.75]
				// [6.7, 7.65, 8.6]
				// [9.55, 10.5, 11.45]
				expectedW := mat.NewDense(4, 3, []float64{1, 1.95, 2.9, 3.85, 4.8, 5.75, 6.7, 7.65, 8.6, 9.55, 10.5, 11.45})
				So(mat.Equal(params["w"], expectedW), ShouldBeTrue)
				// param["b"]
				// [0.2, 1.1, 2]
				expectedB := mat.NewVecDense(3, []float64{0.2, 1.1, 2.0})
				So(mat.Equal(params["b"], expectedB), ShouldBeTrue)
			})
		})
	})
}
