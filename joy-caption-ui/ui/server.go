package ui

import (
	"fmt"
	"joy-caption-ui/process"
	"net/http"
	"strings"
	"time"
)

func Service() {

	comm := make(chan ServerStatus, 1)

	srv := &http.Server{
		Addr:              ":5678",
		ReadTimeout:       500 * time.Second,
		ReadHeaderTimeout: 30 * time.Second,
		WriteTimeout:      120 * time.Second,
		IdleTimeout:       0,
		MaxHeaderBytes:    0,
	}

	go func(srv *http.Server, comm chan ServerStatus) {
		fileService := http.FileServer(http.Dir("public"))
		proc := &process.ProcessParams{}
		fmt.Print("Starting server on port 5678")
		mux := http.NewServeMux()
		mux.HandleFunc("/", func(writer http.ResponseWriter, request *http.Request) {
			if strings.Contains(request.URL.Path, "api/v1/") {
				fmt.Println("process request")
				proc.ServeHTTP(writer, request)
			} else {
				fmt.Println("serving html")
				fileService.ServeHTTP(writer, request)
			}
		})
		srv.Handler = mux
		err := srv.ListenAndServe()
		if err != nil {
			comm <- PANIC
			panic(err)
		}
	}(srv, comm)

	for {
		select {
		case status := <-comm:
			if status == EXIT || status == PANIC {
				err := srv.Close()
				if err != nil {
					panic(err)
				}
				return
			}
		}
	}
}
