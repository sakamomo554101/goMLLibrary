# goMLLibrary

## このリポジトリについて

Go言語でニューラルネットワークに関連したコンポーネントをまとめたものです

## 環境

* Go 1.10

## Build&Run

```Sh
// 必要なOSSをインストールする
$ make dep

// Mnistのサンプルコードを動かす
$ make build-sample
```
## 実装したレイヤー

### Activation

* Relu
* Sigmoid
* Tanh
* SoftmaxWithCrossEntropy

### NeraulNetworkCell

* Affine

### Optimizer

* SGD
