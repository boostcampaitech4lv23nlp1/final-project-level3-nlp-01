import React, { useRef, useState } from "react";
import "./App.css";
import SttPage from "./sttPage";
import SummarizationPage from "./summarizationPage";
import QuestionGenerationPage from "./questionGenerationPage";

function App() {
  const [, updateState] = useState();
  const forceUpdate = () => {
    updateState({});
  };

  const serverUrl = "http://127.0.0.1:30001/";

  const sttServerUrl = "http://115.85.181.112:30001/";
  const postProcessServerUrl = "http://49.50.164.49:30001/";
  const summarizationServerUrl = "http://49.50.162.97:30001/";
  const keywordExtractionServerUrl = "http://101.101.208.64:30001/";
  const questionGenerationServerUrl = "http://118.67.143.178:30001/";

  const [inputList, setInputList] = useState([]);

  let [taskBarList, setTaskBarList] = useState([true, false, false, false]);
  let [saveWavFileFlag, setSaveWavFileFlag] = useState(false);
  let [loddingFlag, setLoddingFlag] = useState(false);
  let [summarizationFlag, setSummarizationFlag] = useState(false);
  let [questionGenerationFlag, setQuestionGenerationFlag] = useState(false);

  const inputRef = useRef();
  let sttStatus = useRef([false, false, false]);
  let postProcessingResult = useRef("");
  let segmentationResult = useRef([]);
  let summarizationResult = useRef("");
  let keywordExtractionResult = useRef([]);
  let questionGenerationResult = useRef([]);

  const onSaveRequest = (task, file) => {
    const formData = new FormData();
    formData.append("file", file);
    
    let url = serverUrl + 'isFileExist';
    
    const sent_data = JSON.stringify({
      'filename': file.name,
      'size': file.size
    })
    console.log(sent_data)
    fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: sent_data
    }).then((response) => response.json()).then((response) => {
        console.log(formData)
        if (response.exist === false) {
          url = serverUrl + 'saveWavFile';
          fetch(url, {
            method: "POST",
            body: formData,
          })
            .then((response) => {
              sttStatus.current = [true, false, false, false];
              setSaveWavFileFlag(true);
              setLoddingFlag(false);
            })
            .catch((error) => {
              console.log(error);
          });      
        } else {
          setSaveWavFileFlag(true);
          setLoddingFlag(false);
        }
      })
    
  };

  // get
  const onSttRequest = () => {
    const url = serverUrl + "speechToText";
    fetch(url, {
      method: "GET",
    })
      .then((response) => response.json())
      .then((response) => {
        if (response.status) {
          sttStatus.current = [true, true, false, false];
        }
        forceUpdate();
      })
      .then(() => {
        onSttPostProcessRequest();
      })
      .catch(() => {
        setLoddingFlag(false);
      });
  };

  const onSttPostProcessRequest = () => {
    const url = serverUrl + "sttPostProcessing";
    fetch(url, {
      method: "GET",
    })
      .then((response) => response.json())
      .then((response) => {
        postProcessingResult.current = response.text;
        console.log(postProcessingResult.current);

        sttStatus.current = [true, true, true, false];
        forceUpdate();
      })
      .then(() => {
        onSegmentationRequest();
      })
      .catch(() => {
        setLoddingFlag(false);
      });
  };

  const onSegmentationRequest = () => {
    const url = serverUrl + "segmentation";
    fetch(url, {
      method: "GET",
    })
      .then((response) => response.json())
      .then((response) => {
        segmentationResult.current = response.text;

        sttStatus.current = [true, true, true, true];
        forceUpdate();
      })
      .then(() => {
        setLoddingFlag(false);
      })
      .catch(() => {
        setLoddingFlag(false);
      });
  };

  const onSummarizationRequest = () => {
    const url = serverUrl + "summarization";
    const sent_data = JSON.stringify({
      stt_output: segmentationResult.current,
    });
    console.log(sent_data);
    fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: sent_data,
    })
      .then((response) => response.json())
      .then((response) => {
        summarizationResult.current = response.summarization_output;
        console.log(summarizationResult.current);
      })
      .then(() => {
        setLoddingFlag(false);
        setSummarizationFlag(true);
        setTaskBarList((prev) => [false, false, true, false]);
      }).then(() => {
        onDownloadSummarizationRequest();
      })
      .catch(() => {
        setLoddingFlag(false);
      });
  };
  
  const onDownloadSummarizationRequest = () => {
    const url = serverUrl + 'summarizationResultDownload';

    const file = inputList[0]
    const sent_data = JSON.stringify({
      'fileName': file.name,
      'size': file.size,
      'result': summarizationResult.current,
    })
    console.log(sent_data);

    fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: sent_data
    }).then((response) => response.blob()).then((blob) => {
      let data = blob;
      let textURL = window.URL.createObjectURL(data);
      let tmpLink = document.createElement('a');
      tmpLink.href = textURL;
      tmpLink.setAttribute('download', 'summarize_export.txt')
      tmpLink.click();
    }).catch((error) => {
      console.log(error);
    })
  }

  const onKeywordExtractionRequest = () => {
    const url = serverUrl + "keyword";
    const sent_data = JSON.stringify({
      seg_docs: segmentationResult.current,
      summary_docs: summarizationResult.current,
    });

    fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: sent_data,
    })
      .then((response) => response.json())
      .then((response) => {
        keywordExtractionResult.current = response.output;
        console.log(keywordExtractionResult.current); // {context: 'something', keyword: Array()}
        forceUpdate();
      })
      .then(() => {
        onQuestionGenerationRequest();
      });
  };

  const onQuestionGenerationRequest = () => {
    const url = serverUrl + "questionGeneration";
    const file = inputList[0]
    const sent_data = JSON.stringify({
      keywords: keywordExtractionResult.current,
      filename: file.name,
      size: file.size
    });

    fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: sent_data,
    })
      .then((response) => response.json())
      .then((response) => {
        questionGenerationResult.current = response.output;
        console.log(questionGenerationResult.current);
      })
      .then(() => {
        setLoddingFlag(false);
        setQuestionGenerationFlag(true);
        setTaskBarList((prev) => [false, false, false, true]);
      });
  };

  return (
    <div className="App">
      <div
        className="text-center text-bg-dark custom-bg"
        style={{ minHeight: "100vh" }}
      >
        <header className="mb-auto">
          <div>
            <nav className="nav nav-masthead justify-content-center py-4">
              <li className="nav-item">
                {taskBarList[0] ? (
                  <a className="nav-link fw-bold py-1 px-2 active" href="#!">
                    intro
                  </a>
                ) : (
                  <a
                    className="nav-link fw-bold py-1 px-2"
                    href="#!"
                    onClick={() => {
                      setTaskBarList((prev) => [true, false, false, false]);
                    }}
                  >
                    intro
                  </a>
                )}
              </li>
              <li className="nav-item">
                {taskBarList[1] ? (
                  <a className="nav-link fw-bold py-1 px-2 active" href="#!">
                    speech to text
                  </a>
                ) : (
                  <a
                    className="nav-link fw-bold py-1 px-2"
                    href="#!"
                    onClick={() => {
                      setTaskBarList((prev) => [false, true, false, false]);
                    }}
                  >
                    speech to text
                  </a>
                )}
              </li>
              <li className="nav-item">
                {taskBarList[2] ? (
                  <a className="nav-link fw-bold py-1 px-2 active" href="#!">
                    summarization
                  </a>
                ) : (
                  <a
                    className="nav-link fw-bold py-1 px-2"
                    href="#!"
                    onClick={() => {
                      if (summarizationFlag) {
                        setTaskBarList((prev) => [false, false, true, false]);
                      } else {
                        alert("?????? ????????? ??????????????????.");
                      }
                    }}
                  >
                    summarization
                  </a>
                )}
              </li>
              <li className="nav-item">
                {taskBarList[3] ? (
                  <a className="nav-link fw-bold py-1 px-2 active" href="#!">
                    question generation
                  </a>
                ) : (
                  <a
                    className="nav-link fw-bold py-1 px-2"
                    href="#!"
                    onClick={() => {
                      if (questionGenerationFlag) {
                        setTaskBarList((prev) => [false, false, false, true]);
                      } else {
                        alert("?????? ????????? ??????????????????.");
                      }
                    }}
                  >
                    question generation
                  </a>
                )}
              </li>
            </nav>
          </div>
        </header>
        <main className="px-3">
          <div>
            {taskBarList[0] ? (
              <div
                className="d-flex justify-content-left align-items-center p-5"
                style={{ minHeight: "85vh", marginLeft: "50px" }}
              >
                <div>
                  <div className="d-flex">
                    <h1 className="display-5 fw-bold py-1">?????? ?????? ?????????</h1>
                  </div>
                  <div className="d-flex">
                    <p className="lead">?????? ?????? ???????????? ?????????</p>
                  </div>
                  <div className="d-flex">
                    <p className="lead">????????? ????????? ??????</p>
                  </div>
                  <div className="d-flex">
                    <p className="lead">
                      ??????????????? ????????? ??????????????? ????????????????????????.
                    </p>
                  </div>
                  <div className="d-flex">
                    <label
                      className="file btn btn-lg btn-light btn-primary fw-bold border-white bg-white mt-3"
                      onClick={() => {
                        setTaskBarList((prev) => [false, true, false, false]);
                      }}
                    >
                      ????????????
                    </label>
                  </div>
                </div>
              </div>
            ) : taskBarList[1] && inputList.length === 0 ? (
              <div
                className="d-flex justify-content-center align-items-center"
                style={{ minHeight: "85vh" }}
              >
                <div>
                  <h1 className="display-5 fw-bold py-1">speech to text</h1>
                  <p className="lead">?????? ????????? ???????????? ?????? ???,</p>
                  <p className="lead">
                    STT ????????? ?????? ???????????? ????????? ??????????????????.
                  </p>
                  <p className="">??? 15??? ?????? ?????? 3??? ????????? ???????????????.</p>
                  <label className="file btn btn-lg btn-light btn-primary fw-bold border-white bg-white mt-3">
                    ?????? ?????????
                    <input
                      type="file"
                      accept="audio/wav"
                      hidden
                      ref={inputRef}
                      onChange={(event) => {
                        if (event.currentTarget.files?.[0]) {
                          const file = event.currentTarget.files[0];
                          setInputList((prev) => [...prev, file]);
                          setLoddingFlag(true);
                          onSaveRequest("saveWavFile", file);
                        }
                      }}
                    />
                  </label>
                </div>
              </div>
            ) : taskBarList[1] && inputList.length !== 0 ? (
              <div
                className="d-flex justify-content-center align-items-center"
                style={{ minHeight: "85vh" }}
              >
                <div>
                  <SttPage
                    file={inputList}
                    segmentationResult={segmentationResult.current}
                  />
                  <div className="d-flex justify-content-center mt-2">
                    {(loddingFlag) && (saveWavFileFlag === false) ? (
                      <label className="btn btn-lg btn-light btn-primary fw-bold border-white bg-white mt-3 disabled">
                        ????????? ???
                      </label>
                    ) : (loddingFlag) && (saveWavFileFlag === true) ? (
                      <label className="btn btn-lg btn-light btn-primary fw-bold border-white bg-white mt-3 disabled">
                        ?????? ???
                      </label>
                    ) : segmentationResult.current.length < 1 ? (
                      <label
                        className="btn btn-lg btn-light btn-primary fw-bold border-white bg-white mt-3"
                        onClick={() => {
                          sttStatus.current = [true, false, false];
                          setLoddingFlag(true);
                          onSttRequest();
                        }}
                      >
                        ?????? ??????
                      </label>
                    ) : (
                      <label
                        className="btn btn-lg btn-light btn-primary fw-bold border-white bg-white mt-3"
                        onClick={() => {
                          setLoddingFlag(true);
                          onSummarizationRequest();
                        }}
                      >
                        ?????? ??????
                      </label>
                    )}
                    {loddingFlag && (
                      <div
                        class="spinner-border text-light"
                        role="status"
                        style={{ marginLeft: "10px", marginTop: "22px" }}
                      >
                        <span class="sr-only"></span>
                      </div>
                    )}
                  </div>
                  <div>
                    {loddingFlag && (
                      <div className="mt-2">
                        {sttStatus.current[0] === false ? (
                          <div></div>
                        ) : sttStatus.current[1] === false ? (
                          <div style={{ fontSize: "1px" }}>
                            (0/2) stt ?????? ???...
                          </div>
                        ) : sttStatus.current[2] === false ? (
                          <div style={{ fontSize: "1px" }}>
                            (1/2) ?????? ?????? ???...
                          </div>
                        ) : sttStatus.current[3] === false ? (
                          <div style={{ fontSize: "1px" }}>
                            (2/2) segment ?????? ???...
                          </div>
                        ) : (
                          <div></div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ) : taskBarList[2] ? (
              <div>
                <div
                  className="d-flex flex-column align-items-center mt-5"
                  style={{ minHeight: "85vh" }}
                >
                  <SummarizationPage
                    summarizationResult={summarizationResult.current}
                  />
                  <div className="d-flex justify-content-center mt-2">
                    {loddingFlag ? (
                      <label className="btn btn-lg btn-light btn-primary fw-bold border-white bg-white mt-3 disabled">
                        ?????? ???
                      </label>
                    ) : (
                      <label
                        className="btn btn-lg btn-light btn-primary fw-bold border-white bg-white mt-3"
                        onClick={() => {
                          // keyward extraction -> question generation
                          setLoddingFlag(true);
                          onKeywordExtractionRequest();
                        }}
                      >
                        ?????? ??????
                      </label>
                    )}
                    {loddingFlag && (
                      <div
                        class="spinner-border text-light"
                        role="status"
                        style={{ marginLeft: "10px", marginTop: "22px" }}
                      >
                        <span class="sr-only"></span>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ) : taskBarList[3] ? (
              <div>
                <div
                  className="d-flex justify-content-center align-items-center"
                  style={{ minHeight: "85vh" }}
                >
                  <QuestionGenerationPage
                    questionGenerationResult={questionGenerationResult.current}
                  />
                </div>
              </div>
            ) : (
              <div>something error</div>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;
