package main

import (
	"embed"
	"joy-caption-ui/joycaption"

	"github.com/wailsapp/wails/v2"
	"github.com/wailsapp/wails/v2/pkg/options"
	"github.com/wailsapp/wails/v2/pkg/options/assetserver"
)

//go:embed all:frontend/dist
var assets embed.FS

func main() {
	// Create an instance of the app structure
	app := NewApp()
	caption := &joycaption.Caption{}

	// Create application with options
	err := wails.Run(&options.App{
		Title:            "joy-caption-ui",
		Width:            640,
		Height:           640,
		BackgroundColour: &options.RGBA{R: 0, G: 0, B: 0, A: 1},
		AssetServer: &assetserver.Options{
			Assets: assets,
		},
		OnStartup: app.startup,
		Bind: []interface{}{
			app,
			caption,
		},
		Debug: options.Debug{
			true,
		},
	})

	if err != nil {
		println("Error:", err.Error())
	}
}
