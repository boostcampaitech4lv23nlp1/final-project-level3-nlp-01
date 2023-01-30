import React, { useRef, useState } from 'react';
import './App.css';
import SttPage from './sttPage';
import SummarizationPage from './summarizationPage';
import QuestionGenerationPage from './questionGenerationPage';

function App() {
    const [, updateState] = useState();
    const forceUpdate = () => {updateState({})}

    const serverUrl = 'http://127.0.0.1:8001/'
    const [inputList, setInputList] = useState([]);

    let [taskBarList, setTaskBarList] = useState([true, false, false, false]);
    let [loddingFlag, setLoddingFlag] = useState(false);
    
    const inputRef = useRef();
    let sttStatus = useRef([false, false, false]);
    let postProcessingResult = useRef("");
    let segmentationResult = useRef([]);
    let summarizationResult = useRef("");
    let keywordExtractionResult = useRef([]);
    let questionGenerationResult = useRef([]);
    

    const onSaveRequest = (task, file) => {
      const formData = new FormData();
      formData.append("file", file)
      for (let value of formData.values()){
        console.log(value)
      }
      const url = serverUrl + task;
      fetch(url, {
        method: 'POST',
        body: formData
      }).then((response) => {
        sttStatus.current = [true, false, false, false];
        setLoddingFlag(false);
      }).catch((error) => {
        console.log(error);
      })
    }

    // get 
    const onSttRequest = () => {
      const url = serverUrl + 'speechToText';
      fetch(url, {
        method: 'GET'
      }).then((response) => response.json())
      .then((response) => {
        if (response.status){
          sttStatus.current = [true, true, false, false];
        }
        forceUpdate();
      }).then(() => {
        onSttPostProcessRequest();
      }).catch(() => {
        setLoddingFlag(false);
      })
    }
    
    const onSttPostProcessRequest = () => {
      const url = serverUrl + 'sttPostProcessing';
      fetch(url, {
        method: 'GET'
      }).then((response) => response.json())
      .then((response) => {
        postProcessingResult.current = response.text;
        console.log(postProcessingResult.current)

        sttStatus.current = [true, true, true, false];
        forceUpdate();
      }).then(() => {
        onSegmentationRequest();
      }).catch(() => {
        setLoddingFlag(false);
      })
    }

    const onSegmentationRequest = () => {
      const url = serverUrl + 'segmentation';
      fetch(url, {
        method: 'GET'
      }).then((response) => response.json())
      .then((response) => {
        segmentationResult.current = response.text;

        sttStatus.current = [true, true, true, true];
        forceUpdate();
      }).then(() => {
        setLoddingFlag(false);
      }).catch(() => {
        setLoddingFlag(false);
      })
    }

    const onSummarizationRequest = () => {
      const url = serverUrl + 'summarization';
      const sent_data = JSON.stringify({
        'stt_output': segmentationResult.current
      })
      console.log(sent_data);
      fetch(url, {
        method: 'POST',
        headers: { 'Content-Type' : 'application/json'},
        body: sent_data
      }).then((response) => response.json())
      .then((response) => {
        summarizationResult.current = response.summarization_output;
        console.log(summarizationResult.current);
      }).then(() => {
        setLoddingFlag(false);
        setTaskBarList((prev) => [false, false, true ,false]);
      }).catch(() => {
        setLoddingFlag(false);
      })
    }

    const onKeywordExtractionRequest = () => {
      const url = serverUrl + 'keyword';
      const sent_data = JSON.stringify({
        'seg_docs': segmentationResult.current,
        'summary_docs': summarizationResult.current
      });
      
      fetch(url,{
        method: 'POST',
        headers: { 'Content-Type' : 'application/json'},
        body: sent_data
      }).then((response) => response.json()).then((response) => {
        keywordExtractionResult.current = response.output;
        console.log(keywordExtractionResult.current); // {context: 'something', keyword: Array()}
        forceUpdate();
      }).then(() => {
        onQuestionGenerationRequest();
      })
    }

    const onQuestionGenerationRequest = () => {
      const url = serverUrl + 'questionGeneration';
      const sent_data = JSON.stringify({
        'keywords': keywordExtractionResult.current
      });

      fetch(url, {
        method: 'POST',
        headers: { 'Content-Type' : 'application/json'},
        body: sent_data
      }).then((response) => response.json())
      .then((response) => {
        questionGenerationResult.current = response.output;
        console.log(questionGenerationResult.current);
      }).then(() => {
        setLoddingFlag(false);
        setTaskBarList((prev) => [false, false, false, true]);
      }).then(() => {
        onDonwloadResultRequest();
      })
    }

    const onDonwloadResultRequest = () => {
      const url = serverUrl + 'downloadResult';
      
      fetch(url, {
        method: 'GET',
      }).then((response) => {
        return response.blob()
      }).then((blob) => {
        let data = blob;
        let csvURL = window.URL.createObjectURL(data);
        let tmpLink = document.createElement('a');
        tmpLink.href = csvURL;
        tmpLink.setAttribute('download', 'export.csv');
        tmpLink.click();
      })
    }

    return (
        <div className='App'>
            <div className='text-center text-bg-dark custom-bg' style={{minHeight: '100vh'}}>
                <header className='mb-auto'>
                    <div>
                        <nav className="nav nav-masthead justify-content-center py-4">
                          <li className="nav-item">
                              {
                                taskBarList[0] ? 
                                <a className="nav-link fw-bold py-1 px-2 active" href="#!">
                                    intro
                                </a>
                                :<a className="nav-link fw-bold py-1 px-2" href="#!" onClick={() => {
                                  setTaskBarList((prev) => [true, false, false, false]);
                                }}>
                                    intro
                                </a>
                              }
                            </li>                          
                            <li className="nav-item">
                              {
                                taskBarList[1] ? 
                                <a className="nav-link fw-bold py-1 px-2 active" href="#!">
                                    speech to text
                                </a>
                                :<a className="nav-link fw-bold py-1 px-2" href="#!" onClick={()=>{
                                  setTaskBarList((prev) => [false, true, false, false]);
                                }}>
                                    speech to text
                                </a>
                              }
                            </li>
                            <li className="nav-item">
                              {
                                taskBarList[2] ?
                                <a className="nav-link fw-bold py-1 px-2 active" href="#!">
                                    summarization
                                </a>
                                :<a className="nav-link fw-bold py-1 px-2" href="#!" onClick={()=>{
                                  setTaskBarList((prev) => [false, false, true, false]);
                                }}>
                                    summarization
                                </a>
                              }
                            </li>
                            <li className="nav-item">
                              {
                                taskBarList[3] ?
                                <a className="nav-link fw-bold py-1 px-2 active" href="#!">
                                    question generation
                                </a>
                                :<a className="nav-link fw-bold py-1 px-2" href="#!" onClick={()=>{
                                  setTaskBarList((prev) => [false, false, false, true]);
                                }}>
                                  question generation
                                </a>
                              }
                            </li>
                        </nav>
                    </div>
                </header>
                <main className='px-3'>
                    <div>{
                        taskBarList[0] ?
                        <div className="d-flex justify-content-left align-items-center p-5" style={{minHeight: '85vh', marginLeft: '50px'}}>
                          <div>
                            <div className="d-flex">
                              <h1 className="display-5 fw-bold py-1">강의 질문 생성기</h1>
                            </div>
                            <div className="d-flex">
                              <p className="lead">강의 음성 데이터를 받으면</p>
                            </div>
                            <div className="d-flex">
                              <p className="lead">일련의 과정을 거쳐</p>
                            </div>
                            <div className="d-flex">
                              <p className="lead">최종적으로 질문을 생성해주는 파이프라인입니다.</p>
                            </div>
                            <div className="d-flex">
                              <label className="file btn btn-lg btn-light btn-primary fw-bold border-white bg-white mt-3" onClick={
                                () => {
                                  setTaskBarList((prev => [false, true, false, false]))
                                }
                              }>
                                시작하기
                              </label>
                            </div>
                          </div>
                        </div>
                        : (taskBarList[1] && inputList.length === 0) ?
                        <div className="d-flex justify-content-center align-items-center" style={{minHeight: '85vh'}}>
                          <div>
                              <h1 className="display-5 fw-bold py-1">speech to text</h1>
                              <p className="lead">음성 파일을 입력으로 받은 후,</p>
                              <p className="lead">STT 모델을 거쳐 텍스트를 결과로 보여드립니다.</p>
                              <label className="file btn btn-lg btn-light btn-primary fw-bold border-white bg-white mt-3">
                                  파일 업로드
                                  <input type="file" accept='audio/wav' hidden ref={inputRef} onChange={
                                      (event) => {
                                        if (event.currentTarget.files?.[0]){
                                          const file = event.currentTarget.files[0];
                                          setInputList((prev => [...prev, file]));
                                          setLoddingFlag(true);
                                          onSaveRequest('saveWavFile', file);
                                        }
                                    }}/>
                              </label>     
                          </div>                          
                        </div>
                        : (taskBarList[1] && inputList.length !== 0) ?
                        <div className="d-flex justify-content-center align-items-center" style={{minHeight: '85vh'}}>
                          <div>
                            <SttPage file={inputList} segmentationResult={segmentationResult.current}/>
                            <div className="d-flex justify-content-center mt-2">
                              {loddingFlag ? 
                              <label className="btn btn-lg btn-light btn-primary fw-bold border-white bg-white mt-3 disabled">
                                작업 중
                              </label>
                              :segmentationResult.current.length < 1 ? <label className="btn btn-lg btn-light btn-primary fw-bold border-white bg-white mt-3" onClick={() => {
                                sttStatus.current = [true, false, false]
                                setLoddingFlag(true);
                                onSttRequest();
                              }}>
                                변환 시작
                              </label>
                              : 
                              <label className="btn btn-lg btn-light btn-primary fw-bold border-white bg-white mt-3" onClick={() => {
                                setLoddingFlag(true);
                                onSummarizationRequest();
                              }}>
                                요약 시작
                              </label>
                              }
                              {loddingFlag && 
                              <div class="spinner-border text-light" role="status" style={{marginLeft: '10px' ,marginTop: '22px'}}>
                                <span class="sr-only"></span>
                              </div>
                              }
                            </div>
                            <div>
                              {
                                loddingFlag && 
                                <div className='mt-2'>
                                {
                                  (sttStatus.current[0] === false) ?
                                  <div></div>
                                  : (sttStatus.current[1] === false) ? 
                                  <div style={{fontSize: '1px'}}>
                                    (0/2) stt 변환 중...
                                  </div>
                                  : (sttStatus.current[2] === false) ? 
                                  <div style={{fontSize: '1px'}}>
                                    (1/2) 교정 작업 중...
                                  </div>
                                  : (sttStatus.current[3] === false) ?
                                  <div style={{fontSize: '1px'}}>
                                    (2/2) segment 작업 중...
                                  </div>
                                  :<div></div>
                                }
                                </div>
                              }
                            </div>
                          </div>
                        </div>
                        : taskBarList[2] ? 
                        <div>
                          <div className="d-flex flex-column align-items-center mt-5" style={{minHeight: '85vh'}}>
                            <SummarizationPage summarizationResult={summarizationResult.current}/>
                            <div className="d-flex justify-content-center mt-2">
                              {
                                loddingFlag ?
                                <label className="btn btn-lg btn-light btn-primary fw-bold border-white bg-white mt-3 disabled">
                                    작업 중
                                </label>  
                                :<label className="btn btn-lg btn-light btn-primary fw-bold border-white bg-white mt-3" onClick={() => {
                                // keyward extraction -> question generation
                                setLoddingFlag(true);
                                onKeywordExtractionRequest();
                              }}>
                                질문 생성
                              </label>
                              }
                              {loddingFlag && 
                                <div class="spinner-border text-light" role="status" style={{marginLeft: '10px' ,marginTop: '22px'}}>
                                  <span class="sr-only"></span>
                                </div>
                              }
                            </div>
                          </div>
                        </div>
                        : taskBarList[3] ? 
                        <div>
                          <div className="d-flex justify-content-center align-items-center" style={{ minHeight: '85vh'}}>
                            <QuestionGenerationPage questionGenerationResult={questionGenerationResult.current}/>
                          </div>
                        </div>
                        : <div>something error</div>
                      }
                    </div>
                </main>
            </div>
        </div>
    );
}

export default App;
