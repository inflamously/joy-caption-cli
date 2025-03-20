package joycaption

import "fmt"

type Caption struct {
}

func (cap *Caption) ApiTransferFiles(files []*FileBuffer) bool {
	for _, file := range files {
		if len(file.Name) <= 0 {
			continue
		}

		//img, format, err := DecodeImageFromBytes(file.Bytes)
		//if err != nil {
		//	panic(err)
		//}

		err := StoreBytesLocally(file.Bytes, file.Name)
		if err != nil {
			fmt.Println(err)
		}
	}

	return true
}
