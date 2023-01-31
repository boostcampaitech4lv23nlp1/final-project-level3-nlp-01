import React, { useState } from "react";

function QuestionGenerationCard(props) {
  const question = props.question;
  const answer = props.answer;
  const [userInput, setUserInput] = useState("");
  const [showFlag, setShowFlag] = useState(false);

  const handleChange = (event) => {
    setUserInput(event.target.value);
  }
  const handleKeyDown = (event) => {
    if (event.key === 'Enter'){
      setShowFlag(true);
    }
  }
  console.log(userInput);

  return (
    <div className="my-5 border-bottom">
      <div className="d-flex flex-column">
        <h3 className="align-self-start">Q.</h3>
        <p className="align-self-start">{question}</p>
        <h3 className="align-self-end">A.</h3>
        <input className="p-1 mb-1" type="text" value={userInput} placeholder="답변을 입력해주세요" style={
          {border:"none", background: "transparent", color: "#a0a0a0", textAlign: 'right'}
        } onChange={handleChange} onKeyDown={handleKeyDown}/>
        {
          (showFlag && (userInput.length > 0)) && (
            <p className="align-self-end">모범 답안은 "{answer}" 입니다.</p>
          )
        }
      </div>
    </div>
  )
}

function QuestionGenerationPage(props) {
  const questionGenerationResult = props.questionGenerationResult;
  
  return (
    <div className="container-md" style={{maxWidth: '80vh'}}>
      <div className="mb-5">
        <h1 className="display-5 fw-bold py-1"> question generation </h1>
        <p className="lead">생성된 질문을 출력합니다.</p>
        
      </div>
      {/* question & answer */}
      {questionGenerationResult.length > 0 && (
          questionGenerationResult.map((d) => (
          <QuestionGenerationCard question={d.question} answer={d.answer}/>)
        )
      )}
      <div className="d-flex justify-content-end m-3" style={{padding: '5px'}}>
        {
        questionGenerationResult.length > 0 ? 
        <label className="btn btn-sm btn-light btn-primary fw-bold border-white">
            download
        </label>
        : <div></div>
        }
      </div>
    </div>
  )
} export default QuestionGenerationPage;