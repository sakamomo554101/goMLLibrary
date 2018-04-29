package model

type LayerType int

const (
	SigmoidType LayerType = iota
	TanhType
	ReluType
	SoftmaxWithLossType
	AffineType
	SgdType
)

type NNModel struct {
	Layers []NNData
}

func NewNNModel() *NNModel {
	nnModel := NNModel{}
	nnModel.Layers = make([]NNData, 0)
	return &nnModel
}

type NNData struct {
	Type      LayerType
	Parameter map[string]NNRawData
}

func NewNNData() NNData {
	data := NNData{}
	data.Parameter = make(map[string]NNRawData)
	return data
}

type NNRawData struct {
	Row     int
	Col     int
	RawData []float64
}
