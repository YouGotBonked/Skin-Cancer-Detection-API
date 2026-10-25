import { Link } from 'react-router';
import axios from 'axios';
import { ChangeEvent, useState, React } from 'react';
import './Upload.css'

const Upload = () => {
    const [selectedFile, setSelectedFile] = useState(null);
    const onFileChange = (event) => {
        setSelectedFile(event.target.files[0]);
    };
    const onFileUpload = () => {
        const formData = new FormData();
        formData.append(
            "myFile",
            selectedFile,
            selectedFile.name
        );
        console.log(selectedFile);
        axios.post("api/uploadfile", formData);
    };
    const fileData = () => {
        if (selectedFile) {
            return (
                <div>
                    <h2>File Details:</h2>
                    <p>File Name: {selectedFile.name}</p>
                    <p>File Type: {selectedFile.type.replaceAll("application/", "")}</p>
                </div>
            );
        } else {
            return (
                <div>
                    <h4>Choose before Pressing the Upload button</h4>
                </div>
            );
        }
    };

    return (
        <div>
            <h1>Upload Your Cells</h1>
            <div>
                <input className='input-file' type="file" onChange={onFileChange} />
                <button onClick={onFileUpload}>Upload!</button>
            </div>
            {fileData()}
        </div>
    );
};

export default Upload;