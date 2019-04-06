package neuralNetwork

import (
	"reflect"
	"testing"

	"github.com/goMLLibrary/core/util"
	. "github.com/smartystreets/goconvey/convey"
)

func TestNewImage(t *testing.T) {
	Convey("Given : 3 * 4の数値が配列で与えられた時", t, func() {
		w := 3
		h := 4
		input := util.CreateFloatArrayByStep(w*h, 0, 1)
		Convey("When : Imageを作成する", func() {
			image := NewImage(input, w, h)
			Convey("Then : 3*4の2次元配列データが出来ていること", func() {
				So(h, ShouldEqual, len(image))
				for i := 0; i < h; i++ {
					So(w, ShouldEqual, len(image[i]))
					So(reflect.DeepEqual(image[i], util.CreateFloatArrayByStep(w, float64(i*w), 1)), ShouldBeTrue)
				}
			})
		})
	})
}

func TestNewImageWithChannel(t *testing.T) {
	Convey("Given : 幅3, 高さ4, チャネル数3の数値が配列で与えられた時", t, func() {
		w := 3
		h := 4
		c := 3
		input := util.CreateFloatArrayByStep(w*h*c, 0, 1)
		Convey("When : ImageWithChannelを作成する", func() {
			imageWithChannel := NewImageWithChannel(input, w, h, c)
			Convey("Then : 3*4*3の3次元配列データが出来ていること", func() {
				So(c, ShouldEqual, len(imageWithChannel))
				for i := 0; i < c; i++ {
					So(h, ShouldEqual, len(imageWithChannel[i]))
					for j := 0; j < h; j++ {
						So(w, ShouldEqual, len(imageWithChannel[i][j]))
						So(reflect.DeepEqual(imageWithChannel[i][j], util.CreateFloatArrayByStep(w, float64(w*(i*h+j)), 1)), ShouldBeTrue)
					}
				}
			})
		})
	})
}

func TestNewImagesWithChannel(t *testing.T) {
	Convey("Given : 幅3, 高さ4, チャネル数3, バッチ数2の数値が配列で与えられた時", t, func() {
		w := 3
		h := 4
		c := 3
		batch := 2
		input := util.CreateFloatArrayByStep(w*h*c*batch, 0, 1)
		Convey("When : ImagesWithChannelを作成する", func() {
			imagesWithChannel := NewImagesWithChannel(input, w, h, c, batch)
			Convey("Then : 3*4*3*2の4次元配列データが出来ていること", func() {
				So(batch, ShouldEqual, len(imagesWithChannel))
				for i := 0; i < batch; i++ {
					So(c, ShouldEqual, len(imagesWithChannel[i]))
					for j := 0; j < c; j++ {
						So(h, ShouldEqual, len(imagesWithChannel[i][j]))
						for k := 0; k < h; k++ {
							So(w, ShouldEqual, len(imagesWithChannel[i][j][k]))
							So(reflect.DeepEqual(imagesWithChannel[i][j][k], util.CreateFloatArrayByStep(w, float64(w*(i*c*h+j*h+k)), 1)), ShouldBeTrue)
						}
					}
				}
			})
		})
	})
}
