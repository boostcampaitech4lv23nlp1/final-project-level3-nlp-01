import { React, useState } from 'react';

function SttPage(props) {
  const inputList = props.file;
  const file = inputList[0];
  const fileName = file.name;
  const [bytePath, setBytePath] = useState("");
  
  const reader = new FileReader();
  reader.readAsDataURL(file);
  reader.onload = function() {
    let bytePath = reader.result;
    setBytePath(bytePath)
  }
  
  return (
    <div>
        <h1 className="display-5 fw-bold py-1">speech to text</h1>
        <p className="lead">{fileName}</p>
        <div>{
          bytePath.length !== 0 && (
            <audio controls>
            <source src={bytePath} type="audio/wav"/>
          </audio>
          )}
        </div>
    </div>
  )
}
export default SttPage;