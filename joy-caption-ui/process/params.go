package process

import (
	"fmt"
	"net/http"
)

type ProcessParams struct {
}

func (p *ProcessParams) ServeHTTP(writer http.ResponseWriter, request *http.Request) {
	//TODO implement me
	fmt.Println("params")
}

func (p *ProcessParams) ExecPath() {
	fmt.Print("Executing folder request")
	// TODO
}
