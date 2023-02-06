if __name__ == "__main__":
    import uvicorn
    import torch
    torch.multiprocessing.set_start_method('spawn', force=True)
    uvicorn.run("app.server:app", host="0.0.0.0", port=30001)