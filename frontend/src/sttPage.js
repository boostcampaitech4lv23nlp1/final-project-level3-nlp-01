import { React, useState } from 'react';

function SegmentationResultView(props) {
  return (
    <div>
      <p className='p-1' style={{textAlign: 'left'}}>
        {props.text}
      </p>
    </div>
  )
}

function SttPage(props) {
  const inputList = props.file;
  const file = inputList[0];
  const fileName = file.name;
  const [bytePath, setBytePath] = useState("");
  const segmentationResult = props.segmentationResult;
  
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
        <div className='mt-3 d-flex justify-content-center'>
        {
          segmentationResult.length > 0 &&
          <div className='d-flex flex-column border overflow-auto p-2' style={{
            minHeight: '50vh', minWidth: '80vh', maxHeight: '50vh', maxWidth: '80vh'
          }}>
            {segmentationResult.map((s) => (<SegmentationResultView text={s}/>))}
          </div>
        }
        </div>
    </div>
  )
}
export default SttPage;