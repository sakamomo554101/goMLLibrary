package util

import (
	. "github.com/smartystreets/goconvey/convey"
	"testing"
)

func TestMaxValue(t *testing.T) {
	Convey("Given : 長さ6の小数点な配列を用意", t, func() {
		values := []float64{-1.2, 1.5, 2, 5.0, 0, 4.2}
		Convey("When : MaxValueを実施", func() {
			k, v := MaxValue(values)
			Convey("Then : キーと値が最大値のキーと値になっていること", func() {
				So(k, ShouldEqual, 3)
				So(v, ShouldEqual, 5.0)
			})
		})
	})
}
