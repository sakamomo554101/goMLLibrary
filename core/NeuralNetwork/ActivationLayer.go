package NeuralNetwork

import (
	"gonum.org/v1/gonum/mat"
	"math"
)

// Sigmoid : シグモイド関数
type Sigmoid struct {
	out mat.Matrix
}

// NewSigmoid : シグモイド関数の素子を取得
func NewSigmoid() *Sigmoid {
	sigmoid := Sigmoid{}
	return &sigmoid
}

func (sigmoid *Sigmoid) Forward(x mat.Matrix) mat.Matrix {
	r, c := x.Dims()
	dense := mat.NewDense(r, c, nil) // zero dense
	dense.Apply(func(i, j int, v float64) float64 {
		return 1.0 / (1.0 + math.Exp(-v))
	}, x)
	sigmoid.out = dense
	return dense
}

func (sigmoid *Sigmoid) Backward(dout mat.Matrix) mat.Matrix {
	r, c := dout.Dims()
	dense := mat.NewDense(r, c, nil) // zero dense
	dense.Apply(func(i, j int, v float64) float64 {
		return v * (1.0 - sigmoid.out.At(i, j)) * sigmoid.out.At(i, j)
	}, dout)
	return dense
}

// Relu : Relu関数
type Relu struct {
	out mat.Matrix
}

// NewRelu : Relu関数の素子を取得
func NewRelu() *Relu {
	r := &Relu{}
	return r
}

func (relu *Relu) Forward(x mat.Matrix) mat.Matrix {
	r, c := x.Dims()
	dense := mat.NewDense(r, c, nil) // zero matrix
	dense.Apply(func(i, j int, v float64) float64 {
		if v > 0 {
			return x.At(i, j)
		} else {
			return 0
		}
	}, x)
	relu.out = dense
	return dense
}

func (relu *Relu) Backward(dout mat.Matrix) mat.Matrix {
	r, c := dout.Dims()
	dense := mat.NewDense(r, c, nil)
	dense.Apply(func(i, j int, v float64) float64 {
		if relu.out.At(i, j) > 0 {
			return v
		} else {
			return 0
		}
	}, dout)
	return dense
}

// Tanh : Tanh関数
type Tanh struct {
	out mat.Matrix
}

// NewTanh : Tanh関数の素子を取得
func NewTanh() *Tanh {
	t := &Tanh{}
	return t
}

func (tanh *Tanh) Forward(x mat.Matrix) mat.Matrix {
	r, c := x.Dims()
	dense := mat.NewDense(r, c, nil)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			dense.Set(i, j, math.Tanh(x.At(i, j)))
		}
	}
	tanh.out = dense
	return dense
}

func (tanh *Tanh) Backward(dout mat.Matrix) mat.Matrix {
	r, c := dout.Dims()
	dense := mat.NewDense(r, c, nil)
	for i := 0; i < c; i++ {
		for j := 0; j < r; j++ {
			dense.Set(i, j, dout.At(i, j)*(1-math.Pow(tanh.out.At(i, j), 2)))
		}
	}
	return dense
}

type SoftmaxWithLoss struct {
	out  mat.Matrix
	t    mat.Matrix
	loss float64
}

func NewSoftmax() *SoftmaxWithLoss {
	s := &SoftmaxWithLoss{}
	return s
}

func (s *SoftmaxWithLoss) Forward(x mat.Matrix, t mat.Matrix) float64 {
	s.t = t
	s.out = s.softmax(x)
	s.loss = s.crossEntropyError(x, t)
	return s.loss
}

func (s *SoftmaxWithLoss) Backward() mat.Matrix {
	r, c := s.t.Dims()
	dense := mat.NewDense(r, c, nil)
	bs := r

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			val := (s.out.At(i, j) - s.t.At(i, j)) / float64(bs)
			dense.Set(i, j, val)
		}
	}
	return dense
}

func (s *SoftmaxWithLoss) softmax(x mat.Matrix) mat.Matrix {
	r, c := x.Dims()
	tmp := mat.NewDense(r, c, nil)
	tmp.Clone(x)
	dense := mat.NewDense(r, c, nil)
	// 行の計算
	for i := 0; i < r; i++ {
		// 1つのデータのベクトルを取得し、合計値と要素の最大値を計算
		// TODO : 処理に時間がかかるため、リファクタリングする
		v := tmp.RowView(i)
		max := mat.Max(v.T())
		sum := 0.0
		for k := 0; k < c; k++ {
			sum += math.Exp(v.At(k, 0) - max)
		}

		// 各列の値を算出
		for j := 0; j < c; j++ {
			val := math.Exp(v.At(j, 0)-max) / sum
			dense.Set(i, j, val)
		}
	}
	return dense
}

// crossEntropyError : 実値と正解データから交差エントロピー誤差を算出
// x : 入力値, t : 正解データ
func (s *SoftmaxWithLoss) crossEntropyError(x mat.Matrix, t mat.Matrix) float64 {
	xr, xc := x.Dims()
	tr, tc := t.Dims()

	// 実際のデータと正解データの行列の形が同じかを確認
	if xr != tr || xc != tc {
		// TODO エラー対応
		return 0.0
	}

	// バッチサイズを取得(行数)
	bs := xr

	// 各値の交差エントロピーを求め、バッチサイズを考慮して平均を出力
	delta := math.Pow10(-7) // logの中身が0にならないように対応
	dense := mat.NewDense(xr, xc, nil)
	for i := 0; i < xr; i++ {
		for j := 0; j < xc; j++ {
			val := -1 * t.At(i, j) * math.Log(x.At(i, j)+delta)
			dense.Set(i, j, val)
		}
	}
	return mat.Sum(dense) / float64(bs)
}
