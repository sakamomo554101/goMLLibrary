package tvm_wrapper

import (
	"fmt"
	"io/ioutil"
	"runtime"
	"./gotvm"
)

// TvmConfig : TVM Module Configuration
type TvmConfig struct {
	DeviceType int64
}

// NewTvmConfig : Create Object of TvmConfig
func NewTvmConfig() *TvmConfig {
	config := TvmConfig{}
	config.DeviceType = (int64)(gotvm.KDLCPU)
	return &config
}

type moduleInfo struct {
	graphModule *gotvm.Module
	inputShape []int64
	outputShape []int64
}

func newModuleInfo(graphModule *gotvm.Module, inputShape []int64, outputShape []int64) *moduleInfo {
	info := moduleInfo{}
	info.graphModule = graphModule
	info.inputShape = inputShape
	info.outputShape = outputShape
	return &info
}

// ModelParam : model parameters struct to build tvm graph
type ModelParam struct {
	ModelLibPath    string
	ModelJSONPath   string
	ModelParamsPath string
	InputShape      []int64
	OutputShape     []int64
}

// NewModelParam : create instance of model param
func NewModelParam(modelLibPath string, modJsonPath string, modParamsPath string,
	inputShape []int64, outputShape []int64) *ModelParam {
	params := ModelParam{
		modelLibPath,
		modJsonPath,
		modParamsPath,
		inputShape,
		outputShape,
	}
	return &params
}

// DebugStr : get debuggable text of model parameters
func (param *ModelParam) DebugStr() string {
	debugStr := fmt.Sprintf("ModelLibPath : %s\n", param.ModelLibPath)
	debugStr += fmt.Sprintf("ModelJSONPath : %s\n", param.ModelJSONPath)
	debugStr += fmt.Sprintf("ModelParamsPath : %s\n", param.ModelJSONPath)
	debugStr += fmt.Sprintf("InputShape : %v\n", param.InputShape)
	debugStr += fmt.Sprintf("OutputShape : %v\n", param.OutputShape)
	return debugStr
}

// TvmWrapper : TVM wrapper struct to use TVM function
type TvmWrapper struct {
	funcNames []string
	config TvmConfig
}

// NewTvmWrapper : Create TVM wrapper object
func NewTvmWrapper() *TvmWrapper {
	wrapper := TvmWrapper{}
	return &wrapper
}

// Initialize : Initialize TVM wrapper struct to use it
func (wrapper *TvmWrapper) Initialize(config TvmConfig) error {
	defer runtime.GC()

	// display gotvm information
	fmt.Printf("TVM Version   : v%v\n", gotvm.TVMVersion)
	fmt.Printf("DLPACK Version: v%v\n\n", gotvm.DLPackVersion)

	// set configuration
	wrapper.config = config

	// get global function names
	funcNames, err := gotvm.FuncListGlobalNames()
	if err != nil {
		fmt.Print(err.Error())
		return err
	}
	wrapper.funcNames = funcNames
	return nil
}

// LoadModel : Load specified model to get inference model
func (wrapper *TvmWrapper) LoadModel(modelParam *ModelParam) (*moduleInfo, error) {
	defer runtime.GC()

	// debug model parameters
	fmt.Print(modelParam.DebugStr())

	// load module library
	fmt.Print("start to load module library...\n")
	modLibP, err := gotvm.LoadModuleFromFile(modelParam.ModelLibPath)
	if err != nil {
		fmt.Print(err.Error())
		return nil, err
	}

	// read module json file
	fmt.Print("start to read module json file...\n")
	bytes, err := ioutil.ReadFile(modelParam.ModelJSONPath)
	if err != nil {
		fmt.Print(err.Error())
		return nil, err
	}
	modJsonStr := string(bytes)

	// create graph module of tvm
	fmt.Print("start to create graph module of tvm...\n")
	funcp, err := gotvm.GetGlobalFunction("tvm.graph_runtime.create")
	if err != nil {
		fmt.Printf(err.Error())
		return nil, err
	}
	// graph_runtime.create
	// arg[0] : model json text
	// arg[1] : model library
	// arg[2] : device type (ex. KDLCPU, KDLGPU...)
	// arg[3] : device id
	graphrt, err := funcp.Invoke(modJsonStr, modLibP, wrapper.config.DeviceType, (int64)(0))
	if err != nil {
		fmt.Print(err.Error())
		return nil, err
	}
	graphmod := graphrt.AsModule()

	// import params to graph module
	fmt.Print("start to import params to graph module...\n")
	bytes, err = ioutil.ReadFile(modelParam.ModelParamsPath)
	if err != nil {
		fmt.Print(err.Error())
		return nil, err
	}
	funcp, err = graphmod.GetFunction("load_params")
	if err != nil {
		fmt.Print(err.Error())
		return nil, err
	}
	_, err = funcp.Invoke(bytes)
	if err != nil {
		fmt.Print(err.Error())
		return nil, err
	}

	// create module information
	fmt.Print("start to create module information...\n")
	info := newModuleInfo(graphmod, modelParam.InputShape, modelParam.OutputShape)
	return info, nil
}

// Infer : Infer the output data from input
func (wrapper *TvmWrapper) Infer(moduleInfo *moduleInfo, input []float32) ([]float32, error) {
	defer runtime.GC()
	graphmod := moduleInfo.graphModule
	inputShape := moduleInfo.inputShape
	outputShape := moduleInfo.outputShape

	// set input
	funcp, err := graphmod.GetFunction("set_input")
	if err != nil {
		fmt.Print(err.Error())
		return nil, err
	}
	// TODO : use device type
	inputForTvm, err := gotvm.Empty(inputShape, "float32", gotvm.CPU(0))
	if err != nil {
		fmt.Print(err)
		return nil, err
	}
	inputForTvm.CopyFrom(input)
	_, err = funcp.Invoke("input", inputForTvm)
	if err != nil {
		fmt.Print(err.Error())
		return nil, err
	}

	// start to inference function
	funcp, err = graphmod.GetFunction("run")
	if err != nil {
		fmt.Print(err.Error())
		return nil, err
	}
	_, err = funcp.Invoke()
	if err != nil {
		fmt.Print(err.Error())
		return nil, err
	}

	// Allocate output array to receive output data from inference function
	out, err := gotvm.Empty(outputShape)
	if err != nil {
		fmt.Print(err.Error())
		return nil, err
	}

	// get output from inference function
	funcp, err = graphmod.GetFunction("get_output")
	if err != nil {
		fmt.Print(err.Error())
		return nil, err
	}
	_, err = funcp.Invoke(int64(0), out)
	if err != nil {
		fmt.Print(err.Error())
		return nil, err
	}
	outAsSlice, _ := out.AsSlice()
	return outAsSlice.([]float32), nil
}