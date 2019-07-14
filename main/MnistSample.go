package main

import (
	"fmt"
	"os"

	"github.com/goMLLibrary/core/graph"
	"github.com/goMLLibrary/core/mnist"
	"github.com/goMLLibrary/core/neuralNetwork"
)

func main() {
	// ニューラルネットワーク層をまとめるレイヤーの作成
	layers := neuralNetwork.NewDefaultNeuralNetworkLayers()

	// 1層目
	layers.Add(neuralNetwork.NewAffine(28*28, 1000))
	layers.Add(neuralNetwork.NewRelu())

	// 2層目
	layers.Add(neuralNetwork.NewAffine(1000, 1000))
	layers.Add(neuralNetwork.NewRelu())

	// 3層目
	layers.Add(neuralNetwork.NewAffine(1000, 10))

	// MNISTデータセットを格納するためのフォルダを作成
	os.Mkdir("data", 0777)

	// MNISTのデータセットを取得
	train, test, err := mnist.LoadData("data")
	if err != nil {
		fmt.Printf("Can't get mnist train and test data! \n")
		os.Exit(-1)
	}

	// 学習時の各種パラメーターの設定
	batchSize := 100
	iterationCount := 1800
	iteracionCountPerEpoch := int(train.Count() / batchSize)

	// 学習時の様子をグラフに描画するための準備
	graphCreater, err := graph.NewGraphCreater("output")
	if err != nil {
		panic("panic! graph can't be used!")
	}
	param := graph.NewGraphParameter()
	param.Title = "MNIST_Test"
	param.XLabel = "iteration"
	param.YLabel = "accuracy"
	trainPoints := graph.NewGraphPoints("train")

	// 学習の実行
	for i := 0; i < iterationCount; i++ {
		// 入力用のデータを取得
		rawSet := mnist.ExtractRandomDataSet(train, batchSize)
		x, t := mnist.ConvertMatrixFromDataSet(rawSet)

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

		// グラフへ追加
		trainPoints.AddPoint(graph.NewPoint(float64(i), acc))
	}

	/*
		iteracionCountPerEpoch = int(test.Count() / batchSize)
		iterationCount = 100
		for i := 0; i < iterationCount; i++ {
			rawSet := mnist.ExtractRandomDataSet(test, batchSize)
			x, t := mnist.ConvertMatrixFromDataSet(rawSet)

			// Forward処理の実施
			loss, acc := layers.Forward(x, t)

			fmt.Printf("test %d iteration : loss is %f, accuracy is %f\n", i, loss, acc)
		}
	*/
	// 予測の実行
	x, t := mnist.ConvertMatrixFromDataSet(test)
	loss, acc := layers.Forward(x, t)
	fmt.Printf("test : loss is %f, accuracy is %f\n", loss, acc)

	// グラフの作成
	graphCreater.SaveLineGraph(param, []graph.GraphPoints{trainPoints})

	os.Exit(0)
}
