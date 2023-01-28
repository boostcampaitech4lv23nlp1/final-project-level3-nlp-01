import React from "react";

function SummarizationResultView(props) {
  return (
    <div>
      <p className='p-1' style={{textAlign: 'left'}}>
        {props.text}
      </p>
    </div>
  )
}


function SummarizationPage(props) {
  const summarizationResult = props.summarizationResult;  

  return (
    <div>
      <h1 className="display-5 fw-bold py-1">summarization</h1>
      <p className="lead">요약이 모두 완료되었습니다.</p>
      <p className="lead">질문을 생성하고 싶다면 "질문 생성" 버튼을 눌러주세요.</p>
      <div className="d-flex flex-column border overflow-auto p-2" style={{
        minHeight: '50vh', minWidth: '80vh', maxHeight: '50vh', maxWidth: '80vh'
      }}>{
        summarizationResult.length > 0 && (
          summarizationResult.map((s) => (<SummarizationResultView text={s}/>))
        )
      }
      </div>
    </div>
  )
} export default SummarizationPage;

