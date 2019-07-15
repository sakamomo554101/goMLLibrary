package model

import (
	"os"
	"reflect"
	"testing"

	"github.com/goMLLibrary/core/neuralNetwork"
	"github.com/goMLLibrary/core/util"
	. "github.com/smartystreets/goconvey/convey"
	"gonum.org/v1/gonum/mat"
)

func TestModelHandler(t *testing.T) {
	Convey("Given : 2層のニューラルネットワークの情報が与えられた時", t, func() {
		nnLayers := neuralNetwork.NewDefaultNeuralNetworkLayers()

		modelPath := "model.db"
		defer os.Remove(modelPath)

		Convey("AND : 1層目：5*10のAffineレイヤーを作成", nil)
		inputSize := 5
		outputSize := 10
		affine := neuralNetwork.NewAffine(inputSize, outputSize)

		Convey("AND : 1層目：Affineレイヤーのパラメーターを初期化し設定", nil)
		w := mat.NewDense(inputSize, outputSize, util.CreateFloatArrayByStep(inputSize*outputSize, 0, 0.5))
		b := mat.NewVecDense(outputSize, util.CreateFloatArrayByStep(outputSize, 0, 1))
		params := make(map[string]mat.Matrix, 2)
		params["w"] = w
		params["b"] = b
		affine.UpdateParams(params)

		Convey("AND : 1層目：Tanhレイヤーを作成", nil)
		tanh := neuralNetwork.NewTanh()

		Convey("AND : 1層目のレイヤーを追加", nil)
		nnLayers.Add(affine)
		nnLayers.Add(tanh)

		Convey("AND : 2層目：10*3のAffineレイヤーを作成", nil)
		inputSize = 10
		outputSize = 3
		affine2 := neuralNetwork.NewAffine(inputSize, outputSize)

		Convey("AND : 2層目：Affineレイヤーのパラメーターを初期化し設定", nil)
		w2 := mat.NewDense(inputSize, outputSize, util.CreateFloatArrayByStep(inputSize*outputSize, 0, 0.5))
		b2 := mat.NewVecDense(outputSize, util.CreateFloatArrayByStep(outputSize, 0, 1))
		params2 := make(map[string]mat.Matrix, 2)
		params2["w"] = w2
		params2["b"] = b2
		affine2.UpdateParams(params2)

		Convey("AND : 2層目のレイヤーを追加", nil)
		nnLayers.Add(affine2)

		Convey("When : NNの情報を保存する", func() {
			err := WriteNNLayers(modelPath, nnLayers)
			So(err, ShouldBeNil)

			Convey("AND : NNの情報を復元する", nil)
			reLayers, err := ReadNNLayers(modelPath)
			So(err, ShouldBeNil)

			Convey("Then : 復元したNNのパラメーターが復元前と同一であること", func() {
				beforeLayers := nnLayers.GetLayers()
				afterLayers := reLayers.GetLayers()

				for i, bLayer := range beforeLayers {
					aLayer := afterLayers[i]
					// レイヤーの型が同一であること
					So(reflect.ValueOf(bLayer).Type(), ShouldEqual, reflect.ValueOf(aLayer).Type())
					// レイヤーのパラメーターが同一であることを確認
					switch aLayer.(type) {
					case *neuralNetwork.Affine:
						aAffine := aLayer.(*neuralNetwork.Affine)
						bAffine := bLayer.(*neuralNetwork.Affine)

						aParams := aAffine.GetParams()
						bParams := bAffine.GetParams()

						// パラメーターの比較
						So(mat.Equal(aParams["w"], bParams["w"]), ShouldBeTrue)
						So(mat.Equal(aParams["b"], bParams["b"]), ShouldBeTrue)
					}
				}
			})

			Convey("Then : 復元したNNと復元前のNNで同一結果が出ること", func() {
				input := mat.NewDense(3, 5, util.CreateFloatArrayByStep(15, 1, 1))
				t := mat.NewDense(3, 3, []float64{0.1, 0.1, 0.8, 0.7, 0.2, 0.1, 0.5, 0.3, 0.2})

				// 予測結果のlossとaccuracyが同一になることを確認
				bLoss, bAcc := nnLayers.Forward(input, t)
				aLoss, aAcc := reLayers.Forward(input, t)
				So(bLoss, ShouldEqual, aLoss)
				So(bAcc, ShouldEqual, aAcc)
			})
		})
	})
}
