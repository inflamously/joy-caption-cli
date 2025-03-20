package joycaption

import (
	"bytes"
	"errors"
	"fmt"
	"image"
	"image/png"
	"os"
	"path"
)

func DecodeImageFromBytes(fileInBytes []byte) (image.Image, string, error) {
	reader := bytes.NewReader(fileInBytes)

	fmt.Println("decoding image")

	decode, format, err := image.Decode(reader)
	if err != nil {
		return nil, "", err
	}

	fmt.Println("decode success")

	return decode, format, nil
}

func getLocalFilepath() (string, error) {
	fmt.Println("acquiring cwd")

	cwd, err := os.Getwd()
	if err != nil {
		return "", err
	}

	folderpath := path.Join(cwd, "temp")

	fmt.Println("checking file path")

	stat, err := os.Stat(folderpath)
	if err == nil {
		return folderpath, nil
	}

	fmt.Println("validating folder path")

	if os.IsNotExist(err) {
		fmt.Println("creating folder path")

		err = os.Mkdir(folderpath, os.ModePerm)
		if err != nil {
			return "", err
		}

		return folderpath, nil
	}

	if !stat.IsDir() {
		return "", errors.New("path is a file")
	}

	fmt.Println("unknown error")

	return "", err
}

func StoreImageLocally(img image.Image) ([]byte, error) {
	buffer := bytes.Buffer{}
	err := png.Encode(&buffer, img)
	if err != nil {
		return nil, err
	}

	return buffer.Bytes(), nil
}

func StoreBytesLocally(data []byte, name string) error {
	fmt.Println("acquiring filepath")
	filepath, err := getLocalFilepath()
	if err != nil {
		return err
	}
	pathToFile := path.Join(filepath, name)

	fmt.Println("writing image")
	err = os.WriteFile(pathToFile, data, 0755)
	if err != nil {
		return err
	}
	return nil
}
