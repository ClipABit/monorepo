from fastapi import UploadFile
import modal
from fastapi import UploadFile, File, HTTPException
import os

image = modal.Image.debian_slim().pip_install("fastapi[standard]", "python-multipart")
app = modal.App(name="example-basic-web", image=image)


@app.cls()
class WebApp:
    @modal.enter()
    def startup(self):
        from datetime import datetime, timezone

        print("🏁 Starting up!")
        self.start_time = datetime.now(timezone.utc)

    @modal.fastapi_endpoint(docs=True)
    def web(self):
        from datetime import datetime, timezone

        current_time = datetime.now(timezone.utc)
        return {"start_time": self.start_time, "current_time": current_time}

    @modal.fastapi_endpoint(method="POST")
    async def upload(self, file: UploadFile = None):
        if file is None:
            raise HTTPException(status_code=400, detail="No file provided")

        # Read file contents
        contents = await file.read()
        file_size = len(contents)

        # log file details
        print(f"Received file: {file.filename}")
        print(f"Content-Type: {file.content_type}")
        print(f"Size: {file_size} bytes")

        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "size_bytes": file_size,
            "status": "received successfully"
        }

