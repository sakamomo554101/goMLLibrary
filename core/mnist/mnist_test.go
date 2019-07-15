package mnist

import (
	"fmt"
	"image/color"
	"os"
	"testing"

	"github.com/petar/GoMNIST"
)

func TestMnistLoad(t *testing.T) {
	os.Mkdir("data", 0777)
	train, _, err := GoMNIST.Load("data")
	if err != nil {
		t.Fatalf("can't load mnist data! error detail is " + err.Error())
	}

	for _, image := range train.Images {
		for i := 0; i < 28; i++ {
			for j := 0; j < 28; j++ {
				y := image.At(i, j).(color.Gray).Y
				fmt.Printf("%d\n", y)
			}
		}
	}
}
