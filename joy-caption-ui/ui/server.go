package ui

import (
	"fmt"
	"joy-caption-ui/process"
	"net/http"
	"os"
	"time"
)

func Service(comm chan ServerStatus) *http.Server {

	srv := &http.Server{
		Addr:              ":5678",
		ReadTimeout:       500 * time.Second,
		ReadHeaderTimeout: 30 * time.Second,
		WriteTimeout:      120 * time.Second,
		IdleTimeout:       0,
		MaxHeaderBytes:    0,
	}

	go func(srv *http.Server, comm chan ServerStatus) {
		workingDir, err := os.Getwd()
		if err != nil {
			panic(err)
		}
		indexPath := http.Dir(".")
		fmt.Printf("Fileserver at: %s %s", workingDir, indexPath)
		fileService := http.FileServer(indexPath)
		_ = &process.ProcessParams{}
		fmt.Println("Starting server on port 5678")
		mux := http.NewServeMux()
		mux.Handle("/", fileService)
		// TODO: Handle Path due to working dir
		//mux.HandleFunc("/", func(writer http.ResponseWriter, request *http.Request) {
		//	if strings.Contains(request.URL.Path, "api/v1/") {
		//		fmt.Println("process request")
		//		proc.ServeHTTP(writer, request)
		//	} else {
		//		fmt.Println("serving html")
		//		fileService.ServeHTTP(writer, request)
		//	}
		//})
		srv.Handler = mux
		comm <- READY
		err = srv.ListenAndServe()
		if err != nil {
			comm <- PANIC
			panic(err)
		}
	}(srv, comm)

	return srv
}
