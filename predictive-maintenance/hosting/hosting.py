from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

# Upload only deployment folder CONTENTS to ROOT of space
for file in ["app.py", "Dockerfile", "requirements.txt"]:
    api.upload_file(
        path_or_fileobj=f"predictive-maintenance/deployment/{file}",
        path_in_repo=file,  #THIS puts it at root
        repo_id="PratzPrathibha/Predictive-maintenance",
        repo_type="space"
    )
