package util

import (
	. "github.com/smartystreets/goconvey/convey"
	"gonum.org/v1/gonum/mat"
	"testing"
)

func TestRawValues(t *testing.T) {
	Convey("Given : 3*3の行列を作成する", t, func() {
		a := mat.NewDense(3, 3, CreateFloatArrayByStep(9, 1, 1))
		Convey("When : rawValuesの処理を実施", func() {
			rawValues := RawValues(a)
			Convey("Then : 9個のデータを格納した配列が取得できること", func() {
				So(len(rawValues), ShouldEqual, 9)
				for i, v := range rawValues {
					So(i+1, ShouldEqual, v)
				}
			})
		})
	})
}

func TestReshape(t *testing.T) {
	Convey("Given : 4*4の行列を作成する", t, func() {
		a := mat.NewDense(4, 4, CreateFloatArrayByStep(16, 1, 1))
		Convey("When : 2×8の行列に変換するためにreshape処理を実施", func() {
			b := Reshape(a, 2, 8)
			Convey("Then : 2×8の行列になっているおり、各要素の値が意図した値になっていること", func() {
				r, c := b.Dims()
				So(r, ShouldEqual, 2)
				So(c, ShouldEqual, 8)
				dense := mat.DenseCopyOf(b)
				for i := 0; i < r; i++ {
					for j, v := range dense.RawRowView(i) {
						So(v, ShouldEqual, i*c+j+1)
					}
				}
			})
		})
	})
}
