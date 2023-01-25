import React, { useRef, useState } from 'react';
import './App.css';
import SttPage from './sttPage';
import PostProcessingPage from './postProcessingPage';

function App() {
    const serverUrl = 'http://127.0.0.1:8000/'
    const inputRef = useRef();
    const [inputList, setInputList] = useState([]);
    let [taskBarList, setTaskBarList] = useState([true, false, false, false, false])
    let [loddingFlag, setLoddingFlag] = useState(false);
    
    
    const onSaveRequest = (task, file) => {
      const formData = new FormData();
      formData.append("file", file)
      for (let value of formData.values()){
        console.log(value)
      }
      const url = serverUrl + task
      fetch(url, {
        method: 'POST',
        body: formData
      }).then((response) => {
        setLoddingFlag(false);
      }).catch((error) => {
        console.log(error);
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
                                <a className="nav-link fw-bold py-1 px-2 active" href="#">
                                    intro
                                </a>
                                :<a className="nav-link fw-bold py-1 px-2" href="#" onClick={() => {
                                  setTaskBarList((prev) => [true, false, false, false, false]);
                                }}>
                                    intro
                                </a>
                              }
                            </li>                          
                            <li className="nav-item">
                              {
                                taskBarList[1] ? 
                                <a className="nav-link fw-bold py-1 px-2 active" href="#">
                                    speech to text
                                </a>
                                :<a className="nav-link fw-bold py-1 px-2" href="#" onClick={()=>{
                                  setTaskBarList((prev) => [false, true, false, false, false]);
                                }}>
                                    speech to text
                                </a>
                              }
                            </li>
                            <li className="nav-item">
                              {
                                taskBarList[2] ?
                                <a className="nav-link fw-bold py-1 px-2 active" href="#">
                                    post processing
                                </a>
                                :<a className="nav-link fw-bold py-1 px-2" href="#" onClick={()=>{
                                  setTaskBarList((prev) => [false, false, true, false, false]);
                                }}>
                                    post processing
                                </a>
                              }
                            </li>
                            <li className="nav-item">
                              {
                                taskBarList[3] ?
                                <a className="nav-link fw-bold py-1 px-2 active" href="#">
                                    feature extraction
                                </a>
                                :<a className="nav-link fw-bold py-1 px-2" href="#" onClick={()=>{
                                  setTaskBarList((prev) => [false, false, false, true, false]);
                                }}>
                                  feature extraction
                                </a>
                              }
                            </li>
                            <li className="nav-item">
                                {
                                  taskBarList[4] ?
                                  <a className="nav-link fw-bold py-1 px-2 active" href="#">
                                    question generation
                                  </a>
                                  :<a className="nav-link fw-bold py-1 px-2" href="#" onClick={()=>{
                                    setTaskBarList((prev) => [false, false, false, false, true]);
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
                                  setTaskBarList((prev => [false, true, false, false, false]))
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
                            <SttPage file={inputList}/>
                            <div className="d-flex justify-content-center mt-2">
                              {loddingFlag ? 
                              <label className="btn btn-lg btn-light btn-primary fw-bold border-white bg-white mt-3 disabled">
                                변환 시작
                              </label>
                              :<label className="btn btn-lg btn-light btn-primary fw-bold border-white bg-white mt-3" onClick={() => {
                                const url = serverUrl + 'speechToText';
                                setLoddingFlag(true);
                                fetch(url, {
                                  method: 'GET',
                                }).then((response) => {
                                  setLoddingFlag(false);
                                  setTaskBarList((prev => [false, false, true, false, false]));
                                }).catch((error) => {
                                  console.log(error)
                                })
                              }}>
                                변환 시작
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
                        : taskBarList[2] ? 
                        <div>
                          <div className="d-flex flex-column align-items-center mt-5" style={{minHeight: '85vh'}}>
                            <PostProcessingPage/>
                            <label className="btn btn-lg btn-light btn-primary fw-bold border-white bg-white mt-3" onClick={() => {
                              const url = serverUrl + 'postProcessing';
                              fetch(url, {
                                method: 'GET',
                              }).then((response) => {
                                console.log(response)
                                setTaskBarList((prev => [false, false, false, true, false]))
                              }).catch((error) => {
                                console.log(error)
                              })
                            }}>
                              후처리
                            </label>
                          </div>
                        </div>
                        : taskBarList[3] ? 
                        <div>
                          
                        </div>
                        : taskBarList[4] ? 
                        <div>

                        </div>
                        : <div>something error</div>
                      }
                    </div>
                    <div>
                      
                    </div>
                </main>
            </div>
        </div>
    );
}

export default App;
