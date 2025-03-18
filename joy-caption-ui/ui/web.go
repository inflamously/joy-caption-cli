package ui

import (
	"os"
	"path"
)

func CheckAssets() bool {
	workingDir, err := os.Getwd()
	if err != nil {
		return false
	}
	assetsPath := path.Join(workingDir, "public")
	_, err = os.Stat(assetsPath)
	if err != nil {
		return false
	}
	return true
}
