import React from "react";

function PostProcessingPage() {

    return (
      <div>
        <h1 className="display-5 fw-bold py-1">post processing</h1>
        <p className="lead">speech to text 결과물을 후처리합니다.</p>
        <div className="d-flex border overflow-auto" style={{
          minHeight: '50vh', minWidth: '80vh', maxHeight: '50vh', maxWidth: '80vh'
        }}>
          <p className="text-left p-3">
            None
          </p>
        </div>
      </div>

    )
} export default PostProcessingPage;

