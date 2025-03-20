import './App.css';
import CaptionForm, {CaptionProcessProps} from "./caption-form";
import {useCallback} from "react";
import {ApiTransferFiles} from "../wailsjs/go/joycaption/Caption";
import {joycaption} from "../wailsjs/go/models";
import FileBuffer = joycaption.FileBuffer;

function App() {
    const transferFiles = useCallback((props: CaptionProcessProps) => {
        console.log("Transfering files!");

        Promise.all(props.files.map(async (file) => {
            const buffer = await file.arrayBuffer()
            return {
                filename: file.name,
                buffer
            }
        })).then((files) => {
            const filesAsUint8 = files.map((file) => ({
                filename: file.filename, data: new Uint8Array(file.buffer)
            }));
            ApiTransferFiles(filesAsUint8.map((fileAsUint8) => {
                return new FileBuffer({
                    bytes: Array.from(fileAsUint8.data),
                    name: fileAsUint8.filename,
                })
            })).catch(console.error);
        })
    }, []);

    return (
        <div id="App">
            <section>
                <CaptionForm onSubmit={transferFiles}/>
            </section>
        </div>
    )
}

export default App
