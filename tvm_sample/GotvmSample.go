package main

import (
	"./tvm_wrapper"
	"fmt"
	"math/rand"
	"os"
	"path"
	"time"
)

func main() {
	fmt.Printf("start\n")

	// get tvm_wrapper object
	wrapper := tvm_wrapper.NewTvmWrapper()
	config := tvm_wrapper.NewTvmConfig()
	err := wrapper.Initialize(*config)
	if err != nil {
		fmt.Printf("%s\n", err.Error())
		return
	}

	// setup model path
	exe, _ := os.Executable()
	prjPath := path.Dir(path.Dir(exe))
	outputPath := path.Join(prjPath, "python", "output")
	modelLibPath := path.Join(outputPath, "resnet50.so")
	modelJSONPath := path.Join(outputPath, "resnet50.json")
	modelParamPath := path.Join(outputPath, "resnet50.params")

	// setup input/output shape to depend on model
	inputShape := []int64{1, 3, 224, 224}
	outputShape := []int64{1, 1000}

	// load model to tvm wrapper object
	modelParam := tvm_wrapper.NewModelParam(modelLibPath, modelJSONPath, modelParamPath, inputShape, outputShape)
	moduleInfo, err := wrapper.LoadModel(modelParam)
	if err != nil {
		fmt.Printf("%s\n", err.Error())
		return
	}

	// inference
	input := randomFloatArray32(0.0, 1.0, (3 * 224 * 224))
	output, err := wrapper.Infer(moduleInfo, input)
	if err != nil {
		fmt.Printf("%s\n", err.Error())
		return
	}
	fmt.Printf("output : %v\n", output)
	fmt.Printf("end\n")
}

// TODO : use CustomRandom
func randomFloat32(r *rand.Rand, min, max float32) float32 {
	return r.Float32()*(max-min) + min
}

// TODO : use CustomRandom
func randomFloatArray32(min, max float32, count int) []float32 {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	list := make([]float32, 0, count)
	for i := 0; i < count; i++ {
		v := randomFloat32(r, min, max)
		list = append(list, v)
	}
	return list
}
