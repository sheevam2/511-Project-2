#!/bin/bash
# Script to copy part3 folder from VM to local machine

VM_USER="shivamp6"
VM_HOST="fa25-cs511-063.cs.illinois.edu"

# Common repo paths to try
REPO_PATHS=(
    "~/cs511-fall2025-p2-shivamp6"
    "~/511-Project-2"
    "~/cs511-fall2025-p2-shivamp6/part3"
)

echo "This script will copy part3 folder from your VM to local."
echo ""
echo "VM: ${VM_USER}@${VM_HOST}"
echo ""

# Ask user for the path
read -p "Enter the path to your repo on VM (or press Enter to try common paths): " VM_REPO_PATH

if [ -z "$VM_REPO_PATH" ]; then
    echo "Trying common paths..."
    for path in "${REPO_PATHS[@]}"; do
        echo "Attempting: ${VM_USER}@${VM_HOST}:${path}/part3"
        rsync -avz --progress "${VM_USER}@${VM_HOST}:${path}/part3/" ./part3/ 2>&1
        if [ $? -eq 0 ]; then
            echo "✓ Successfully copied from ${path}/part3"
            exit 0
        fi
    done
    echo "❌ Could not find part3 folder. Please specify the exact path."
    exit 1
else
    echo "Copying from: ${VM_USER}@${VM_HOST}:${VM_REPO_PATH}/part3"
    rsync -avz --progress "${VM_USER}@${VM_HOST}:${VM_REPO_PATH}/part3/" ./part3/
    if [ $? -eq 0 ]; then
        echo "✓ Successfully copied part3 folder"
    else
        echo "❌ Copy failed. Please check the path and try again."
        exit 1
    fi
fi

