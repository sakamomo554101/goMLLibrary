package mnist

import (
	"fmt"
	"image"
	"image/color"
	"path"

	"github.com/goMLLibrary/core/util"
	"github.com/petar/GoMNIST"
	"gonum.org/v1/gonum/mat"
)

var downloadMaps = map[string]string{
	"train-images-idx3-ubyte.gz": "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
	"train-labels-idx1-ubyte.gz": "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
	"t10k-images-idx3-ubyte.gz":  "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
	"t10k-labels-idx1-ubyte.gz":  "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
}

// LoadData : Mnistのデータセットを取得
func LoadData(rootPath string) (trainSet *MnistDataSet, testSet *MnistDataSet, err error) {
	// mnistデータがなければ、ダウンロードする
	downloadMnistDataIfNeeded(rootPath)

	// train : Mnistの学習用データ
	// test : Mnistのテスト用データ
	train, test, err := GoMNIST.Load(rootPath)
	if err != nil {
		return nil, nil, err
	}

	trainDataSet := newMnistDataSet(train)
	testDataSet := newMnistDataSet(test)
	return trainDataSet, testDataSet, nil
}

type MnistDataSet struct {
	dataSet []MnistData
	nCol    int
	nRow    int
}

func newMnistDataSet(set *GoMNIST.Set) *MnistDataSet {
	dataSet := MnistDataSet{nCol: set.NCol, nRow: set.NRow}
	dataSet.dataSet = make([]MnistData, 0, set.Count())
	for i, rawData := range set.Images {
		data := newMnistDataFromGoMNISTData(rawData, uint8(set.Labels[i]))
		dataSet.addData(data)
	}
	return &dataSet
}

// ExtractRandomDataSet : 指定したmnistのデータセットからランダムに指定サイズ分だけのデータを抽出する
func ExtractRandomDataSet(rawSet *MnistDataSet, count int) *MnistDataSet {
	dataSet := MnistDataSet{nCol: rawSet.nCol, nRow: rawSet.nRow}
	dataSet.dataSet = make([]MnistData, 0, count)
	if rawSet.Count() < count {
		panic("count is not match!")
	}
	randomIndexs := util.RandomIntArray(rawSet.Count(), count)
	for _, index := range randomIndexs {
		dataSet.dataSet = append(dataSet.dataSet, rawSet.GetData(index))
	}
	return &dataSet
}

func ConvertMatrixFromDataSet(rawSet *MnistDataSet) (x mat.Matrix, labels mat.Matrix) {
	row := rawSet.Count()
	col := rawSet.nRow * rawSet.nCol
	xDense := mat.NewDense(row, col, nil)
	tDense := mat.NewDense(row, 10, nil)
	for i, mnistData := range rawSet.GetDataSet() {
		xDense.SetRow(i, mat.VecDenseCopyOf(mnistData.GetImageVector()).RawVector().Data)
		tDense.SetRow(i, mat.VecDenseCopyOf(mnistData.GetLabelVector()).RawVector().Data)
	}
	return xDense, tDense
}

func (set *MnistDataSet) addData(data MnistData) {
	set.dataSet = append(set.dataSet, data)
}

func (set *MnistDataSet) GetData(i int) MnistData {
	return set.dataSet[i]
}

func (set *MnistDataSet) GetDataSet() []MnistData {
	return set.dataSet
}

func (set *MnistDataSet) Count() int {
	return len(set.dataSet)
}

func (set *MnistDataSet) GetNCol() int {
	return set.nCol
}

func (set *MnistDataSet) GetNRow() int {
	return set.nRow
}

type MnistData struct {
	rawImage image.Image
	label    uint8
}

func newMnistDataFromGoMNISTData(src GoMNIST.RawImage, label uint8) MnistData {
	data := MnistData{src, label}
	return data
}

func (data *MnistData) GetImageVector() mat.Vector {
	if data.rawImage.ColorModel() != color.GrayModel {
		panic("mnist data is not gray model!")
	}
	if data.rawImage.Bounds().Min.X != 0 || data.rawImage.Bounds().Min.Y != 0 {
		panic("mnist data size is not match!")
	}

	r := data.rawImage.Bounds().Max.X * data.rawImage.Bounds().Max.Y

	vec := mat.NewVecDense(r, nil)
	for i := 0; i < data.rawImage.Bounds().Max.Y; i++ {
		for j := 0; j < data.rawImage.Bounds().Max.X; j++ {
			index := j + data.rawImage.Bounds().Max.X*i
			vec.SetVec(index, float64(data.rawImage.At(j, i).(color.Gray).Y))
		}
	}
	return vec
}

func (data *MnistData) GetLabelVector() mat.Vector {
	vec := mat.NewVecDense(10, nil)
	vec.SetVec(int(data.label), 1)
	return vec
}

func downloadMnistDataIfNeeded(rootPath string) error {
	// download mnist files
	for key, v := range downloadMaps {
		filePath := path.Join(rootPath, key)

		// if mnist file is not found, download it.
		if !util.Exists(filePath) {
			err := util.DownloadFile(filePath, v)
			if err != nil {
				fmt.Printf("%s \n", err.Error())
				return err
			}
		}
	}
	return nil
}
