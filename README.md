# Taro Leaf Blight Classification

This project classifies a taro leaf image as Healthy, Early Blight, Mid Blight, or Late Blight. The classifier uses an ImageNet-pretrained MobileNetV2 feature extractor and a custom head trained on the [Project-AgML taro blight stage dataset](https://huggingface.co/datasets/Project-AgML/taro_blight_stage_classification).

## Live AWS deployment

| Resource | Link | What to expect |
| --- | --- | --- |
| Dashboard | [Open the live dashboard](https://find-identity-generated-acm.trycloudflare.com/) | Predict a leaf stage, view model uptime and dataset insights, upload labelled images, and monitor retraining. |
| Swagger UI | [Open the live API docs](https://find-identity-generated-acm.trycloudflare.com/docs) | Explore the FastAPI routes and run requests with **Try it out**. |
| Model health | [Open the live health endpoint](https://find-identity-generated-acm.trycloudflare.com/health) | Check `status`, `model_loaded`, `model_version`, and `uptime_seconds` as JSON. |

Open the dashboard and enter **`grader`** in the browser's username field. Use the password supplied separately. The dashboard, Swagger, and health endpoint share the same HTTPS host and login. To check health from a terminal, this command prompts for the password:

```bash
curl --fail --user grader https://find-identity-generated-acm.trycloudflare.com/health
```

The service runs on AWS EC2 and is reachable independently of the developer's computer. The HTTPS address is a Cloudflare Quick Tunnel and remains valid while its EC2 tunnel container is running. Recreating or restarting the tunnel generates a new address.

### Try the live workflow

1. **Predict a leaf stage.** Open **Prediction**, choose [the bundled Mid Blight sample](load_tests/sample.jpg), and select **Run prediction**. The dashboard displays the predicted class, confidence, inference time, and class probabilities. During AWS verification the baseline model classified this example as **Mid Blight** with **96.5% confidence**. The active retrained model may produce different probabilities.
2. **Inspect the training data.** Open **Retraining**. The deployed database was populated with eight real leaf images, two per class. On a fresh instance, choose the matching class and use **Save images** for each folder under [`load_tests/samples/`](load_tests/samples/). The API validates image format and deduplicates by SHA-256. It rejects the same image under conflicting labels.
3. **Trigger retraining.** Select **Start retraining**. The API starts a background job and the dashboard polls for its status. The job reads the saved image bytes from SQLite, applies the same image preprocessing as prediction, and fine-tunes the repository's `.keras` checkpoint.
4. **Verify activation.** After the job completes, check the new version in the dashboard or the live health endpoint. The verified AWS run reported **87.5% training accuracy** after three epochs on eight images. Use the held-out split to measure generalization.

## Local setup

Clone the repository and open its root:

```bash
git clone https://github.com/Akinloye3264/Summative-assignment-MLOP.git
cd Summative-assignment-MLOP
```

Docker Compose runs the full dashboard and API on port `8080`. Native Python is also available for development and testing.

### Run with native Python

Use Python 3.12 (64-bit). On Windows PowerShell, from the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
.\.venv\Scripts\python.exe -m uvicorn api.main:app --env-file .env
```

On Linux or macOS use `python3.12 -m venv .venv`, `cp .env.example .env`, and `.venv/bin/python` in place of the Windows venv executable. Use a platform with a wheel for the pinned [TensorFlow CPU build](https://pypi.org/project/tensorflow-cpu/2.21.0/#files).

Open the [model dashboard](http://127.0.0.1:8000/) or the [interactive API documentation](http://127.0.0.1:8000/docs). FastAPI serves the dashboard, so a separate frontend process is unnecessary. The models and class metadata are included in Git, so API startup does not require the original image dataset. To run the tests, install `requirements-dev.txt` into the same environment and run `python -m pytest` with that environment's Python.

| Action | Endpoint |
| --- | --- |
| Read process uptime and model version | `GET /health` |
| Inspect active model metadata | `GET /model/info` |
| Read saved offline evaluation metrics | `GET /metrics` |
| Predict one JPEG, PNG, or WebP | `POST /predict` with multipart field `file` |
| Upload labelled images | `POST /datasets/upload` with `label` and repeated `files` fields |
| Check labelled upload totals | `GET /datasets/summary` |
| Start a background retraining job | `POST /retrain` |
| Poll a job for its status and training metrics | `GET /retrain/{job_id}` |

Upload one batch per class using `Early_Blight`, `Healthy`, `Late_Blight`, or `Mid_Blight`. The current defaults require at least two distinct images per class and eight total. Uploads are stored as image bytes, labels, and SHA-256 digests in `data/uploaded/uploads.sqlite3`. Job records and the promoted model are written under `data/retraining/`.

## Training and evaluation

The [training notebook](notebook/taro_leaf_blight_classifier.ipynb) was run in Google Colab with a GPU and mounts `MyDrive/MLOps_Taro_Leaf_Blight` for its outputs. It acquires the dataset, inspects image features, creates stratified splits, preprocesses images, trains the head, fine-tunes MobileNetV2, evaluates the held-out test split, and exports the inference model. The training checkpoint is `models/taro_leaf_blight_model.keras`, and the inference model is `models/taro_leaf_blight_model.h5`.

Use `requirements-training.txt` in a separate training environment. A local notebook run also needs Jupyter and a replacement for the Colab Drive mount and absolute output paths. The original dataset download is about 1.27 GB. The notebook also writes `taro_model_artifacts.zip` under its Drive project folder, which may contain a recoverable copy of the training outputs.

The saved evaluation used 2,738 test images. It reports 69.72% accuracy, 0.7407 cross-entropy loss, 66.21% macro precision, 65.74% macro recall, 64.72% macro F1, and 92.59% top-2 accuracy. Mid Blight had 47.23% recall. See the [evaluation metrics](models/metrics.json) and [per-class report](models/classification_report.csv).

The [class distribution](reports/figures/class_distribution.png) shows that Healthy images substantially outnumber Late Blight images. The [feature comparison](reports/figures/feature_comparison.png) shows that Late Blight images have the lowest green-pixel proportion and highest edge density. Early Blight has the highest mean brightness, and Mid Blight has the highest mean saturation. The distributions overlap, and backgrounds can influence all four measurements.

## Run locally with Docker Compose

On Windows, update to a supported Windows 11 release. Docker Desktop currently lists build 22631 as the minimum on Windows 11 and WSL 2.1.5 as the minimum WSL version. Use Windows Update and restart, then run `wsl --install --no-distribution` in administrator PowerShell if WSL 2 is missing, followed by `wsl --update`. Restart if prompted. Check `wsl --version`, then [install Docker Desktop](https://docs.docker.com/desktop/setup/install/windows-install/) and select the WSL 2 backend. On a Linux VM, use [Docker Engine and the Compose plugin](https://docs.docker.com/compose/install/).

When Docker is running, use the repository root:

```powershell
docker compose version
docker compose up --build --detach --wait --wait-timeout 600
curl.exe --fail http://127.0.0.1:8080/health
```

Open the [containerized dashboard](http://127.0.0.1:8080/) or [Swagger UI](http://127.0.0.1:8080/docs). Both use port 8080. The image includes the frontend and both model formats. The Compose volume persists uploads, retraining jobs and the promoted checkpoint across container replacements. Split upload batches that would exceed the proxy's 100 MB request limit.

The default settings support the demo. Copy [`.env.example`](.env.example) to `.env` if you need to change them. Compose accepts `APP_PORT` for the host port and the following training settings:

| Setting | Default |
| --- | ---: |
| `RETRAIN_MIN_IMAGES` | 8 |
| `RETRAIN_MIN_PER_CLASS` | 2 |
| `RETRAIN_EPOCHS` | 3 |
| `MAX_UPLOAD_FILES` | 50 |

Use `docker compose logs -f api` to watch the API and training logs. Use `docker compose down` to stop the local stack and retain its volume.

### Demonstrate database-backed retraining

1. Rebuild using `docker compose up --build --detach --wait`. On its first startup, SQLite imports existing images from the old class folders in the Compose volume. It keeps the original files as a backup and uses SQLite for subsequent reads and writes.
2. Open the dashboard's **Retraining** section. On a fresh deployment, upload the two photos in each `load_tests/samples/` class folder using its matching label. On an upgraded deployment, the existing images should already appear in the upload counts. Re-uploading identical images with the same label counts them as duplicates.
3. Check the saved rows and byte counts from PowerShell:

   ```powershell
   docker compose exec -T api python -c "import sqlite3; db=sqlite3.connect('/app/data/uploaded/uploads.sqlite3'); print(db.execute('SELECT label, COUNT(*), SUM(length(content)) FROM images GROUP BY label').fetchall())"
   ```

4. Click **Start retraining**. The job snapshots the uploaded image IDs, reads their bytes from SQLite, and preprocesses them for fine-tuning. Wait for the dashboard to report that the new model is active. Confirm the new version at [`/health`](http://127.0.0.1:8080/health).
5. Describe the job's accuracy as training accuracy. Use labelled held-out images to evaluate model quality.

## AWS deployment setup

The cloud deployment uses one x86-64 Ubuntu EC2 instance and one API replica. FastAPI serves the dashboard and API together. Local Nginx routes requests to FastAPI, and a password-protected Nginx gateway sits in front of the stack for the Cloudflare HTTPS tunnel. The Compose volume stores the SQLite database, job records, and promoted checkpoint on the EC2 disk.

1. Launch an Ubuntu x86-64 EC2 instance with enough memory for TensorFlow. The deployment uses a `t3.small` with 2 GiB of RAM. Enable SSH for your IP and connect with your EC2 key.
2. Install Docker Engine and Compose on Ubuntu. [Docker installation guide](https://docs.docker.com/engine/install/ubuntu/)
3. Clone the repository on the instance and build the stack:

   ```bash
   git clone https://github.com/Akinloye3264/Summative-assignment-MLOP.git
   cd Summative-assignment-MLOP
   sudo docker compose up --build --detach --wait --wait-timeout 600
   curl --fail http://127.0.0.1:8080/health
   ```

4. Keep the Compose proxy on loopback. Verify the dashboard privately from your PC by forwarding an unused local port, for example `ssh -N -L 8081:127.0.0.1:8080 -i PATH_TO_KEY ubuntu@EC2_HOST`, and opening [localhost:8081](http://127.0.0.1:8081/).
5. For HTTPS access, configure a loopback gateway with HTTP Basic authentication and password hashes stored outside the repository. Forward the gateway to `127.0.0.1:8080` and point Cloudflare Tunnel at the gateway. Verify that unauthenticated requests get HTTP 401, then sign in and check the dashboard, Swagger, and health. The current link is listed at the top of this README.

## Locust prediction benchmark

Install `requirements-dev.txt` into the Python 3.12 environment. By default, the workload cycles through eight images in `load_tests/samples/`, two per class. It posts each image to `/predict` and checks that successful responses contain a predicted class. Set `LOCUST_IMAGE` to use a single image instead. The images come from the [Project-AgML taro blight dataset](https://huggingface.co/datasets/Project-AgML/taro_blight_stage_classification), credited to [Yinka-Banjo et al.](https://doi.org/10.17632/3knm93dkc5.2) under CC BY 4.0.

```powershell
New-Item -ItemType Directory -Force reports/locust | Out-Null
docker compose up --detach --wait --wait-timeout 600 --scale api=1
.\.venv\Scripts\python.exe -m locust -f load_tests/locustfile.py --headless --host http://127.0.0.1:8080 -u 20 -r 5 -t 60s --stop-timeout 20 --csv reports/locust/one-replica --html reports/locust/one-replica.html
docker compose up --detach --wait --wait-timeout 600 --scale api=2
.\.venv\Scripts\python.exe -m locust -f load_tests/locustfile.py --headless --host http://127.0.0.1:8080 -u 20 -r 5 -t 60s --stop-timeout 20 --csv reports/locust/two-replicas --html reports/locust/two-replicas.html
```

Use the final `POST /predict` row of each HTML report. Record request count, failure rate, requests per second, median and 95th-percentile response times. The CSV files provide periodic snapshots. Run both tests on the same host with the same image and no retraining job running.

On November 21, 2026, both recorded runs used only `load_tests/sample.jpg` (row 4, labelled Mid Blight). To repeat that payload on PowerShell, set `$env:LOCUST_IMAGE = 'load_tests/sample.jpg'` before both runs. The runs used the same Docker Desktop engine with 4 CPUs and about 7.7 GiB RAM, 20 users, a spawn rate of 5 users per second, four warm-up predictions, and a 60-second time limit. The second replica's logs confirmed that it served prediction requests.

| API replicas | Requests | Failures | Requests/sec | Median | p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| [One](reports/locust/docker-20261121-065033-api-1.html) | 386 | 0 | 6.28 | 2,600 ms | 3,200 ms |
| [Two](reports/locust/docker-20261121-065033-api-2.html) | 485 | 0 | 7.84 | 1,800 ms | 3,300 ms |

Two replicas increased throughput by about 25% and reduced median latency by about 31% in this run. The p95 was 100 ms higher.

A native Windows smoke test with one Locust user for 10 seconds completed 14 requests with zero failures, 180 ms median response time and 360 ms p95. To reproduce it, use `--host http://127.0.0.1:8000` and `-u 1 -r 1 -t 10s` with a separate report name.

## Demo and submission

- **Camera-on video:** YouTube link pending. Show a labelled prediction, an upload or the SQLite records, retraining progress, the activated model version, and the two Locust reports.
- **Evaluation:** The notebook contains the original held-out evaluation. The production prediction smoke test and AWS retraining workflow passed. A full held-out comparison of the original and retrained checkpoints is pending. `GET /metrics` returns the original offline evaluation.
- **Submission:** Submit the repository ZIP on the first attempt and the [GitHub repository URL](https://github.com/Akinloye3264/Summative-assignment-MLOP) on the second. Share the grader password privately.
