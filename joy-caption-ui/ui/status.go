package ui

type ServerStatus = int

const (
	PANIC = ServerStatus(iota)
	EXIT
)
