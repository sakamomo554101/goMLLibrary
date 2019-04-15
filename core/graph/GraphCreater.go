package graph

import (
	"os"
	"path/filepath"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

type GraphCreater struct {
	p      *plot.Plot
	outDir string
}

func NewGraphCreater(outputDir string) (*GraphCreater, error) {
	gc := GraphCreater{}
	p, err := plot.New()
	if err != nil {
		return nil, err
	}
	gc.p = p

	// グラフ情報の保存先フォルダの確認（無ければ作成）
	gc.outDir = outputDir
	if _, err := os.Stat(outputDir); err != nil {
		os.Mkdir(outputDir, 0777)
	}

	return &gc, nil
}

// SaveGraph : 指定した座標情報で直線グラフを作成し、画像で保存する
func (gc *GraphCreater) SaveLineGraph(parameter GraphParameter, pointsList []GraphPoints) error {
	gc.p.Title.Text = parameter.Title
	gc.p.X.Label.Text = parameter.XLabel
	gc.p.Y.Label.Text = parameter.YLabel

	for _, points := range pointsList {
		if err := plotutil.AddLinePoints(gc.p, points.key, points.convertPlotterXYs()); err != nil {
			return err
		}
	}

	err := gc.p.Save(vg.Length(parameter.Width), vg.Length(parameter.Height), gc.createFilePath(parameter))
	return err
}

func (gc *GraphCreater) createFilePath(parameter GraphParameter) string {
	return filepath.Join(gc.outDir, parameter.Title+".png")
}

// GraphParameter : グラフ表示時の各種パラメーターを指定
type GraphParameter struct {
	Title  string
	XLabel string
	YLabel string
	Width  float64
	Height float64
}

func NewGraphParameter() GraphParameter {
	param := GraphParameter{Width: float64(4 * vg.Inch), Height: float64(4 * vg.Inch)}
	return param
}

// Point : 座標（xy座標）
type Point struct {
	X float64
	Y float64
}

func NewPoint(x float64, y float64) Point {
	p := Point{x, y}
	return p
}

// GraphPoints : グラフの座標情報を格納した構造体
type GraphPoints struct {
	points []Point
	key    string
}

func NewGraphPoints(key string) GraphPoints {
	gp := GraphPoints{key: key}
	points := make([]Point, 0)
	gp.points = points
	return gp
}

func (gp *GraphPoints) AddPoint(p Point) {
	gp.points = append(gp.points, p)
}

func (gp *GraphPoints) convertPlotterXYs() plotter.XYs {
	pts := make(plotter.XYs, len(gp.points))
	for i := range pts {
		pts[i].X = gp.points[i].X
		pts[i].Y = gp.points[i].Y
	}
	return pts
}
