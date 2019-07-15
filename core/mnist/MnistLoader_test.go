package mnist

import (
	"os"
	"testing"

	. "github.com/smartystreets/goconvey/convey"
)

func TestMnistLoader(t *testing.T) {
	Convey("Given : Mnistのデータを取得", t, func() {
		os.Mkdir("data", 0777)
		train, test, err := LoadData("data")

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
