package main

import (
	"fmt"
	"./tvm_wrapper"
	"os"
	"path"
)

func main() {
	fmt.Printf("start\n")

	// get tvm_wrapper object
	wrapper := tvm_wrapper.NewTvmWrapper()
	config := tvm_wrapper.NewTvmConfig()
	wrapper.Initialize(*config)

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
	wrapper.LoadModel(modelParam)

	fmt.Printf("end\n")
}
