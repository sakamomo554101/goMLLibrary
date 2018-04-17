package Sample

import (
	"fmt"
	"github.com/goMLLibrary/core/Mnist"
	"github.com/goMLLibrary/core/NeuralNetwork"
	. "github.com/smartystreets/goconvey/convey"
	"testing"
)

func TestMnist(t *testing.T) {
	Convey("Given : 5層のフリーコネクトなニューラルネットワークを作成", t, func() {
		// ニューラルネットワーク層をまとめるレイヤーの作成
		layers := NeuralNetwork.NewNeuralNetworkLayers()

		/*// 1層目
		// 入力はMnistの画像サイズが28 * 28に合わせる
		layers.Add(NeuralNetwork.NewAffine(28*28, 1200))
		layers.Add(NeuralNetwork.NewRelu())

		// 2層目
		layers.Add(NeuralNetwork.NewAffine(1200, 600))
		layers.Add(NeuralNetwork.NewRelu())

		// 3層目
		layers.Add(NeuralNetwork.NewAffine(600, 300))
		layers.Add(NeuralNetwork.NewRelu())

		// 4層目
		layers.Add(NeuralNetwork.NewAffine(300, 100))
		layers.Add(NeuralNetwork.NewRelu())

		// 5層目
		// 出力は0 - 9（10クラス）に合わせる
		layers.Add(NeuralNetwork.NewAffine(100, 10))*/

		layers.Add(NeuralNetwork.NewAffine(28*28, 1000))
		layers.Add(NeuralNetwork.NewRelu())
		layers.Add(NeuralNetwork.NewAffine(1000, 1000))
		layers.Add(NeuralNetwork.NewRelu())
		layers.Add(NeuralNetwork.NewAffine(1000, 10))

		Convey("AND : MNISTのデータセットを取得", nil)
		train, test, err := Mnist.LoadData()
		So(err, ShouldBeNil)

		Convey("When : 学習処理を実施", func() {
			Convey("AND : 学習時のパラメーターを設定", nil)
			batchSize := 100
			iterationCount := 6000
			iteracionCountPerEpoch := int(train.Count() / batchSize)

			for i := 0; i < iterationCount; i++ {
				// 入力用のデータを取得
				rawSet := Mnist.ExtractRandomDataSet(train, batchSize)
				x, t := Mnist.ConvertMatrixFromDataSet(rawSet)

				// forward処理の実施
				loss, acc := layers.Forward(x, t)

				// backward処理の実施
				layers.Backward()

				// 1epoch事にloss,accuracyを出力
				if (i % iteracionCountPerEpoch) == 0 {
					fmt.Printf("train %d iteration : loss is %f, accuracy is %f\n", i, loss, acc)
				}

				// 勾配のupdate処理の実施
				layers.Update()
			}

			Convey("Then : テストデータで予測処理を実施", func() {
				iteracionCountPerEpoch = int(test.Count() / batchSize)
				iterationCount = 1000
				for i := 0; i < iterationCount; i++ {
					rawSet := Mnist.ExtractRandomDataSet(test, batchSize)
					x, t := Mnist.ConvertMatrixFromDataSet(rawSet)

					// Forward処理の実施
					loss, acc := layers.Forward(x, t)

					fmt.Printf("test %d iteration : loss is %f, accuracy is %f\n", i, loss, acc)
				}
			})
		})

	})

}
