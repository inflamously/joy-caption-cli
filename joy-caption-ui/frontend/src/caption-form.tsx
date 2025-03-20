import {ChangeEvent, FormEvent, useCallback, useState} from "react";

export type CaptionProcessProps = {
    files: File[];
}

const FieldsetPath = (props: {
    onFileChange: (files: File[]) => void;
}) => {
    const {onFileChange} = props;
    const handleFileSubmit = useCallback((ev: ChangeEvent<HTMLInputElement>) => {
        if (ev.target.files && ev.target.files.length > 0) {
            onFileChange?.(Array.from(ev.target.files))
        }
    }, []);

    return <fieldset>
        <label htmlFor="path">Path</label>
        <input id="path" className="folder-selection" multiple={true} title="" type="file" onChange={handleFileSubmit}/>
    </fieldset>
}

const FieldsetCaptionSettings = () => {
    return <fieldset>
        <label htmlFor="description">Describe</label>
        <select id="description">
            <option>Descriptive</option>
            <option>Describe</option>
            <option>Descriptive (Informal)</option>
            <option>Training Prompt</option>
            <option>Training Flux Prompt (Old)</option>
            <option>Training Flux Prompt</option>
            <option>MidJourney</option>
            <option>Booru tag list</option>
            <option>Booru-like tag list</option>
            <option>Art Critic</option>
            <option>Product Listing</option>
            <option>Social Media Post</option>
        </select>
        <label htmlFor="length">Length</label>
        <select id="length">
            <option>Short</option>
            <option>Medium</option>
            <option>Long</option>
        </select>
        <label htmlFor="folder">Folder</label>
    </fieldset>
}

const FieldsetCustomPrompt = () => {
    return <fieldset>
        <label htmlFor="custom-prompt">Custom Prompt</label>
        <input id="custom-prompt" type="text"/>
    </fieldset>
}

const FieldsetProcessSettings = () => {
    return <fieldset>
        <label htmlFor="batch-size">Batch size</label>
        <input
            id="batch-size"
            type="number"
            min="1"
            max="4"
            value="1"/>
    </fieldset>
}

const CaptionForm = (props: {
    onSubmit: (props: CaptionProcessProps) => void,
}) => {
    const {onSubmit} = props

    const [files, setFiles] = useState<File[]>([])

    const startCaptionProcess = useCallback(() => {
        onSubmit({
            files,
        })
    }, [onSubmit, files])

    const handleSubmit = useCallback((ev: FormEvent<HTMLFormElement>) => {
        ev.preventDefault()
    }, [])

    return <form onSubmit={handleSubmit} id="joy-caption-form">
        <FieldsetPath onFileChange={setFiles}/>
        <FieldsetCaptionSettings/>
        <FieldsetCustomPrompt/>
        <FieldsetProcessSettings/>
        <button type={"submit"} onClick={startCaptionProcess}>Submit</button>
    </form>
}

export default CaptionForm