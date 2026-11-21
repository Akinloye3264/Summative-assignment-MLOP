# Taro Leaf Blight Classification

This project classifies a taro leaf image as Healthy, Early Blight, Mid Blight, or Late Blight. The classifier uses an ImageNet-pretrained MobileNetV2 feature extractor and a custom head trained on the [Project-AgML taro blight stage dataset](https://huggingface.co/datasets/Project-AgML/taro_blight_stage_classification).

## Run the API locally

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
| Predict one JPEG, PNG, or WebP | `POST /predict` with multipart field `file` |
| Upload labelled images | `POST /datasets/upload` with `label` and repeated `files` fields |
| Check labelled upload totals | `GET /datasets/summary` |
| Start a background retraining job | `POST /retrain` |
| Poll a job for its status and training metrics | `GET /retrain/{job_id}` |

Upload one batch per class using `Early_Blight`, `Healthy`, `Late_Blight`, or `Mid_Blight`. The current defaults require at least two distinct images per class and eight total. Once the dashboard shows enough images, click **Start retraining** and follow the job status. Uploads are stored in class directories under `data/uploaded/`. Job records and the promoted model are written under `data/retraining/`.

## Training and evaluation

The [training notebook](notebook/taro_leaf_blight_classifier.ipynb) was run in Google Colab with a GPU and mounts `MyDrive/MLOps_Taro_Leaf_Blight` for its outputs. It acquires the dataset, inspects image features, creates stratified splits, preprocesses images, trains the head, fine-tunes MobileNetV2, evaluates the held-out test split, and exports the inference model. The training checkpoint is `models/taro_leaf_blight_model.keras`, and the inference model is `models/taro_leaf_blight_model.h5`.

Use `requirements-training.txt` in a separate training environment. A local notebook run also needs Jupyter and a replacement for the Colab Drive mount and absolute output paths. The original dataset download is about 1.27 GB. The notebook also writes `taro_model_artifacts.zip` under its Drive project folder, which may contain a recoverable copy of the training outputs.

The saved evaluation used 2,738 test images. It reports 69.72% accuracy, 0.7407 cross-entropy loss, 66.21% macro precision, 65.74% macro recall, 64.72% macro F1, and 92.59% top-2 accuracy. Mid Blight had 47.23% recall. See the [evaluation metrics](models/metrics.json) and [per-class report](models/classification_report.csv).

The [class distribution](reports/figures/class_distribution.png) shows that Healthy images substantially outnumber Late Blight images. The [feature comparison](reports/figures/feature_comparison.png) shows that Late Blight images have the lowest green-pixel proportion and highest edge density. Early Blight has the highest mean brightness, and Mid Blight has the highest mean saturation. The distributions overlap, and backgrounds can influence all four measurements.

## Docker and deployment

On Windows, update to a supported Windows 11 release. Docker Desktop currently lists build 22631 as the minimum on Windows 11 and WSL 2.1.5 as the minimum WSL version. Use Windows Update and restart, then run `wsl --install --no-distribution` in administrator PowerShell if WSL 2 is missing, followed by `wsl --update`. Restart if prompted. Check `wsl --version`, then [install Docker Desktop](https://docs.docker.com/desktop/setup/install/windows-install/) and select the WSL 2 backend. On a Linux VM, use [Docker Engine and the Compose plugin](https://docs.docker.com/compose/install/).

When Docker is running, use the repository root:

```powershell
docker compose version
docker compose up --build --detach --wait --wait-timeout 600
curl.exe --fail http://127.0.0.1:8080/health
```

Open the [containerized dashboard](http://127.0.0.1:8080/) or [Swagger UI](http://127.0.0.1:8080/docs). Both use port 8080. The image includes the frontend and both model formats. The Compose volume persists uploads, retraining jobs and the promoted checkpoint across container replacements. Split upload batches that would exceed the proxy's 100 MB request limit.

For a cloud demonstration, use a Linux x86-64 VM and the same Compose commands. The proxy listens on the VM's loopback interface. Forward the port to your machine with `ssh -L 8080:127.0.0.1:8080 user@vm`, then open the same dashboard URL locally. For a public submission URL, put an HTTPS gateway in front of the proxy and require authentication for upload and retraining access. Use one API replica for the retraining demo and scale up for the prediction benchmark.

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

## Assignment status

- Add database-backed upload storage for the retraining rubric. The current code uses files for images and JSON for job records.
- Run a retraining demonstration with labelled leaf photos and evaluate the resulting model on held-out data before promoting it. An isolated TensorFlow run with eight synthetic uploads verified that the trained model could be promoted and loaded. Prediction and retraining use the same aspect-ratio-preserving preprocessing. The current retraining job records training metrics, and `GET /metrics` returns the original offline evaluation.
- Deploy to a cloud host with persistent storage and evaluate the deployed API using labelled held-out images. The Docker build and local prediction smoke test passed.
- Add the deployed URL and camera-on YouTube demo after recording both prediction and retraining. Submit the repository ZIP on the first attempt and the [GitHub repository URL](https://github.com/Akinloye3264/Summative-assignment-MLOP) on the second.
