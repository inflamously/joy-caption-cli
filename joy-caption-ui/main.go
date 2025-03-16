package main

import (
	"joy-caption-ui/ui"
)

func main() {
	comm := make(chan ui.ServerStatus)
	srv := ui.Service(comm)

	for {
		select {
		case status := <-comm:
			if status == ui.EXIT || status == ui.PANIC {
				err := srv.Close()
				if err != nil {
					panic(err)
				}
			}

			if status == ui.READY {
				ui.Run()
			}
		}
	}
}
