package neuralNetwork

import (
	"testing"

	"github.com/goMLLibrary/core/util"
	. "github.com/smartystreets/goconvey/convey"
	"gonum.org/v1/gonum/mat"
)

func TestAffine(t *testing.T) {
	Convey("Given : アフィン変換のレイヤーが一つ与えられた時", t, func() {
		Convey("AND : 重みが3*2行列(in = 3, out = 2), 初期値は1-6とする", nil)
		w := mat.NewDense(3, 2, util.CreateFloatArrayByStep(6, 1, 1))
		Convey("AND : バイアスが2次元, 初期値は-2,-1とする", nil)
		b := mat.NewVecDense(2, []float64{-2, -1})
		aff := newAffine(w, b)
		Convey("When : 入力xを2*3行列とし、値を5-10とする", func() {
			x := mat.NewDense(2, 3, util.CreateFloatArrayByStep(6, 5, 1))
			out := aff.Forward(x)
			Convey("Then : Forward処理を行う. 2*2の行列が出力される", func() {
				// out
				// [56, 75]
				// [83, 111]
				expectedOut := mat.NewDense(2, 2, []float64{56, 75, 83, 111})
				So(mat.Equal(out, expectedOut), ShouldBeTrue)
			})

			Convey("AND : 誤差doutを2*2行列とし、値を5-20とする", nil)
			dout := mat.NewDense(2, 2, util.CreateFloatArrayByStep(4, 5, 5))
			Convey("Then : Backward処理を行う. 2*2の行列が出力される", func() {
				dx := aff.Backward(dout)
				// dx
				// [25, 55, 85]
				// [55, 125, 195]
				expectedDx := mat.NewDense(2, 3, []float64{25, 55, 85, 55, 125, 195})
				So(mat.Equal(dx, expectedDx), ShouldBeTrue)
				// dw
				// [145, 210]
				// [165, 240]
				// [185, 270]
				expectedDw := mat.NewDense(3, 2, []float64{145, 210, 165, 240, 185, 270})
				So(mat.Equal(aff.dw, expectedDw), ShouldBeTrue)
				// db
				// [20, 30]
				expectedDb := mat.NewVecDense(2, []float64{20, 30})
				So(mat.Equal(aff.db, expectedDb), ShouldBeTrue)
			})

			Convey("AND : 各値にして修正", nil)
			params := make(map[string]mat.Matrix)
			params["w"] = mat.NewDense(3, 2, util.CreateFloatArrayByStep(6, 10, 10))
			params["b"] = mat.NewVecDense(2, []float64{2, 4})
			Convey("Then : Update処理を行う.", func() {
				aff.UpdateParams(params)
				ps := aff.GetParams()
				So(mat.Equal(ps["w"], params["w"]), ShouldBeTrue)
				So(mat.Equal(ps["b"], params["b"]), ShouldBeTrue)
			})
		})
	})
}
