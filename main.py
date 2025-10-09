import os
import datetime
import FastAPI
from contextlib import asynccontextmanager

model_path = "/models"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once at startup. Cleans up on shutdown.
    """
    os.makedirs(model_path, exist_ok=True)
   
    yield  # Wait for app shutdown

    #More code here


app = FastAPI(lifespan=lifespan)


# Endpoint for version information
@app.get("/version")
async def version():
    s = Settings()
    return {
        "version": s.version,
    }


# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": str(datetime.datetime.now())}


# Start the server
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7200)
