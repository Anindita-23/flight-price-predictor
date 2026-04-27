# Flight Price Predictor - Setup & Deployment Guide

This document covers how to manage the dataset using DVC and how to deploy the web application to an AWS EC2 instance using Docker.

---

## 1. DVC Setup (Data Version Control)

We use DVC to track the large `data/flight_price_data.csv` file by backing it up to Google Drive. The `.pkl` model files are tracked normally by Git for easier deployment.

### A. Initializing DVC locally
If you clone this repository on a new machine and need to download the CSV data to retrain the model, run these commands:

```bash
# 1. Configure your local Google Drive OAuth credentials
dvc remote modify --local gdrive gdrive_client_id "605587833912-0jtd1fe9t46o5e6sct0tcdfkd9e361mr.apps.googleusercontent.com"
dvc remote modify --local gdrive gdrive_client_secret "YOUR_CLIENT_SECRET"

# 2. Download the data from Google Drive
dvc pull
```
*(When you run `dvc pull`, a browser window will open asking you to log into your Google Account to authenticate).*

### B. Uploading new data
If you add a new CSV file or modify the existing one and want to push it to Drive:
```bash
dvc add data/flight_price_data.csv
dvc push
```

---

## 2. AWS EC2 Deployment Guide

The easiest way to host this website for free is using an AWS EC2 `t2.micro` instance (available in the AWS Free Tier). Because the model (`.pkl`) files are tracked in Git, the deployment process is extremely fast.

### Step 1: Launch the EC2 Instance
1. Go to your **AWS Console** -> **EC2** -> **Launch Instance**.
2. Name: `flight-predictor-server`
3. OS: **Ubuntu Server 24.04 LTS**
4. Instance Type: **t2.micro** (Free tier eligible)
5. Key Pair: Create a new key pair (e.g., `aws-key`), download the `.pem` file, and keep it safe.
6. Network Settings: 
   - Check **Allow SSH traffic**
   - Check **Allow HTTP traffic**
   - Check **Allow HTTPS traffic**
7. Click **Launch Instance**.

### Step 2: Open Port 5000 (Firewall)
1. Go to your running EC2 Instances list, click your instance.
2. Click the **Security** tab -> Click the Security Group (`sg-...`).
3. Click **Edit inbound rules** -> **Add rule**.
4. Set Type to **Custom TCP**, Port Range to **5000**, Source to **Anywhere-IPv4** (`0.0.0.0/0`).
5. Save rules.

### Step 3: Connect to your Server
Open a terminal on your computer and SSH into the server using the `.pem` key you downloaded:

*(Note: On Windows, ensure your terminal is in the folder where the `.pem` file is located)*
```bash
ssh -i "aws-key.pem" ubuntu@YOUR_AWS_PUBLIC_IP
```

### Step 4: Install Docker & Git on the Server
Once you are logged into the AWS server terminal, run:
```bash
# Update server
sudo apt update

# Install Docker and Git
sudo apt install docker.io docker-compose-v2 git -y

# Give ubuntu user permission to run Docker commands
sudo usermod -aG docker ubuntu
newgrp docker
```

### Step 5: Clone & Run the Website!
```bash
# Clone your repository
git clone https://github.com/Anindita-23/flight-price-predictor.git
cd flight-price-predictor

# Make sure you are on the correct branch
git checkout feature/dvc-docker

# Start the Flask Web Server using Docker
docker compose up serve -d
```

### Step 6: View your website
Open your browser and navigate to:
`http://YOUR_AWS_PUBLIC_IP:5000`

---

## 3. Local Docker Development

If you want to run or test the Docker containers locally on your own computer:

**To serve the website:**
```bash
docker compose up serve
```
*(Access it at http://localhost:5000)*

**To retrain the model in an isolated container:**
```bash
docker compose --profile train up train
```
*(This will run `train_model.py` and overwrite the `.pkl` files in the `model/` directory with the newly trained models).*
