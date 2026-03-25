from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/health")
def health():
    return "app is healty"




if __name__ == "__main__":
    uvicorn.run()