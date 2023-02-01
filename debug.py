if __name__ == "__main__":
    import uvicorn
    import torch
    torch.multiprocessing.set_start_method('spawn', force=True) 
    uvicorn.run("app.server:app", host="127.0.0.1", port=30001, reload=True)