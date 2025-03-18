package ui

import (
	"fmt"
	"joy-caption-ui/process"
	"net/http"
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
		hasAssets := CheckAssets()
		if !hasAssets {
			panic("<workingDir>/public assets folder must exist, please retry running from correct working directory")
		}

		fileService := http.FileServer(http.Dir("public"))
		proc := &process.ProcessParams{}
		fmt.Println("Starting server on port 5678")
		mux := http.NewServeMux()
		mux.Handle("/", fileService)
		mux.HandleFunc("/api/v1/", proc.ServeHTTP)
		//mux.HandleFunc("/api/v1/", func(writer http.ResponseWriter, request *http.Request) {
		//	if strings.Contains(request.URL.Path, "") {
		//		fmt.Println("process request")
		//		proc.ServeHTTP(writer, request)
		//	} else {
		//		fmt.Println("serving html")
		//		fileService.ServeHTTP(writer, request)
		//	}
		//})
		srv.Handler = mux
		comm <- READY
		err := srv.ListenAndServe()
		if err != nil {
			comm <- PANIC
			panic(err)
		}
	}(srv, comm)

	return srv
}
