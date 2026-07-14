# run.py
import uvicorn

if __name__ == "__main__":
    import os
    # Get the port Render gives us, or default to 8000 for local testing
    port = int(os.environ.get("PORT", 8000))

    # This immediately wakes up the network binding before loading main.py profiles
    uvicorn.run("main:app", host="0.0.0.0", port=port, workers=1)
