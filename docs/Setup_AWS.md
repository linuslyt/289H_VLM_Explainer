# 289H VLM Explainer - Setup & Installation Guide

This guide details the steps to set up the environment, download datasets, configure paths, and run the VLM Explainer application.

## 📋 System Requirements (AWS EC2)

Before starting, ensure you have provisioned an AWS EC2 instance with the following specifications:

- **Operating System:** Amazon Linux 2 or Ubuntu.
- **Storage:** **Minimum 200 GB** General Purpose SSD (gp3).
  - _Note: The COCO dataset and unzipping process require significant space._
- **Instance Type:**
  - **Recommended:** `g4dn.xlarge` or `g5.xlarge` (if GPU acceleration is required for the VLM).
  - **Minimum (CPU only):** `t3.2xlarge` (Expect significantly slower performance).
- **Security Group:** Ensure **Port 22** (SSH) is open. We will tunnel the application port, so Port 8000 does not strictly need to be open to the public internet, but it is good practice to manage your firewall rules.

---

## Part 1: Server Setup

**📍 Run the following commands on your AWS EC2 Instance.**

### 1\. Connect to Server

Replace `path/to/key.pem` and the IP address with your specific details.

```bash
ssh -i ~/.ssh/id_rsa_289h.pem ec2-user@<YOUR_INSTANCE_IP>
```

### 2\. Clone Repository & Install Conda

```bash
# Clone the repository
git clone https://github.com/linuslyt/289H_VLM_Explainer.git
cd 289H_VLM_Explainer/

# Download and Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Follow the prompts (type 'yes' to license and init)

# Refresh shell to load Conda
source ~/.bashrc
```

### 3\. Create Environment & Install Dependencies

```bash
# Create and activate environment
conda create --name xl_vlm python=3.9 -y
conda activate xl_vlm

# Install project in editable mode
pip install -e .

# Install system dependencies
conda install -c bioconda perl-xml-libxml -y
conda install -c conda-forge openjdk -y

# Install evaluation packages
pip install git+https://github.com/bckim92/language-evaluation.git
python -c "import language_evaluation; language_evaluation.download('coco')"
python -c "import nltk; nltk.download('words')"
```

---

## Part 2: Data Setup (MS COCO)

**📍 Run on AWS EC2 Instance.**

This process involves downloading large files (approx 20GB). Ensure your EBS volume is mounted and has space.

```bash
# Go back to home/root for organization
cd ..

# 1. Create directory structure
mkdir -p data/coco
mkdir -p data/coco/karpathy
cd data/coco

# 2. Download MS COCO 2014 Images
wget http://images.cocodataset.org/zips/train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip

# 3. Unzip images
unzip -q train2014.zip
unzip -q val2014.zip

# 4. Download Karpathy Splits (Train/Val/Test JSONs)
wget http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
unzip -q caption_datasets.zip
mv dataset_coco.json karpathy/

# 5. Clean up zip files to reclaim disk space
rm train2014.zip val2014.zip caption_datasets.zip dataset_flickr8k.json dataset_flickr30k.json

# 6. Move dataset to the System Root (Required by the codebase)
cd ../..
# Moves coco/ to /home/ec2-user/coco/
mv data/coco/ coco/
rm -rf data/
```

---

## Part 3: Path Configuration

**📍 Run on AWS EC2 Instance.**

The original codebase contains hardcoded paths specific to previous users (`ytllam`, `mshukor`, etc.). Run the commands below to update all paths to point to your current user (`ec2-user`).

```bash
# Navigate back to git repo
cd ~/289H_VLM_Explainer

# 1. Update general hardcoded paths in all files
grep -rl "/media/data/ytllam" . | while read -r file; do
    sed -i "s|/media/data/ytllam|/home/ec2-user|g" "$file"
done

# 2. Update specific init file
sed -i "s|/home/ytllam|/home/ec2-user|g" src/acronim_server/__init__.py

# 3. Update COCO Data Directory Paths
grep -rl "/media/data/ytllam/coco" . | xargs sed -i 's|/media/data/ytllam/coco|/home/ec2-user/coco|g'
grep -rl "/data/mshukor/data/coco" . | xargs sed -i 's|/data/mshukor/data/coco|/home/ec2-user/coco|g'

# 4. Update Repository Root Paths
grep -rl "/home/ytllam/xai/xl-vlms" . | xargs sed -i 's|/home/ytllam/xai/xl-vlms|/home/ec2-user/289H_VLM_Explainer|g'
grep -rl "/home/khayatan/xl_vlms_cvpr/xl-vlms" . | xargs sed -i 's|/home/khayatan/xl_vlms_cvpr/xl-vlms|/home/ec2-user/289H_VLM_Explainer|g'

echo "Path configuration complete."
```

---

## Part 4: Launching the Backend Server

**📍 Run on AWS EC2 Instance.**

Install final server dependencies and start the backend.

```bash
# Install backend requirements
pip install fastapi uvicorn pydantic python-multipart pillow torch

# Start the server
# This will listen on port 8000
uvicorn main:app --host 0.0.0.0 --port 8000
```

_Leave this terminal window open._

---

## Part 5: Launching the Frontend

**📍 Run this on YOUR LOCAL COMPUTER (Laptop/Desktop)**

Do **not** run these commands on the AWS server. You need a separate terminal window on your local machine.

### 1\. Establish SSH Tunnel

This command connects your local port `8000` to the AWS server's port `8000`. This allows your local React frontend to talk to the remote Python backend.

```bash
# Replace with your specific Key path and AWS IP Address
ssh -N -L 8000:localhost:8000 -i ~/.ssh/id_rsa_289h.pem ec2-user@<YOUR_AWS_IP>
```

_Note: This command will appear to hang or do nothing. This is normal. It means the tunnel is open. Keep this terminal open._

### 2\. Run React Client

Open a **third terminal** on your local machine. Navigate to the project folder (ensure you have the code cloned locally as well, or at least the frontend directory).

_Prerequisite: Ensure you have [Node.js](https://nodejs.org/) installed locally._

```bash
# Navigate to the frontend directory (adjust path as needed)
cd 289H_VLM_Explainer

# Install Node dependencies
npm install

# Start the development server
npm run dev
```

Open your browser to the URL shown in the terminal (usually `http://localhost:3000`) to use the application.
