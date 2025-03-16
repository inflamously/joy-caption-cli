package ui

import (
	"os/exec"
	"runtime"
)

func Run() {
	if runtime.GOOS == "windows" {
		cmd := exec.Command("chrome", "--version")
		if err := cmd.Run(); err != nil {
			panic(err)
		}
	}

	if runtime.GOOS == "darwin" {
		cmd := exec.Command("open", "http://localhost:5678")
		if err := cmd.Run(); err != nil {
			panic(err)
		}
	}
}
