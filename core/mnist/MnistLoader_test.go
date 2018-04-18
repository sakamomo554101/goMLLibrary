package mnist

import (
	. "github.com/smartystreets/goconvey/convey"
	"testing"
)

func TestMnistLoader(t *testing.T) {
	Convey("Given : Mnistのデータを取得", t, func() {
		train, test, err := LoadData()

		Convey("Then : エラーが発生しないこと", func() {
			So(err, ShouldBeNil)
		})

		Convey("Then : 学習データの数が60000件であること. テストデータの数が10000件であること", func() {
			So(train.Count(), ShouldEqual, 60000)
			So(test.Count(), ShouldEqual, 10000)
		})
	})
}

/*func TestMnistData(t *testing.T) {
	train, test, err := LoadData()
	if err != nil {
		t.Errorf("Load mnist data is failed.")
	}

	// 学習データの中身を確認

}*/
