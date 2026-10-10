# Orange Fruit Disease Detection — Deployment Starter

## 1. Add your trained model
Copy the trained **YOLOv8 classification** checkpoint from your existing project to this folder and name it:

`best.pt`

According to your comparison notebook, the referenced original Windows path is:

`C:\Users\omkar\Fruit_Disease_Project\yolo_results\yolov8_classification\weights\best.pt`

Do not use the object-detection checkpoint for this image-level classification app.

## 2. Run locally
Open a terminal in this folder and run:

```bash
python -m venv .venv
```

Activate the environment, then:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 3. Deploy on Streamlit Community Cloud
1. Create a GitHub repository for this app.
2. Upload `app.py`, `requirements.txt`, and `best.pt`.
3. In Streamlit Community Cloud, create a new app and select that repository and `app.py`.

Note: if the model file is too large for a normal GitHub repository, use Git LFS or another supported model-hosting option and update the app to download it securely. Do not commit private credentials.

## Dataset classes
- blackspotted_orange
- cankered_orange
- fresh_orange
- green_orange
- rotten_orange
