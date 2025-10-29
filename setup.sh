#!/bin/bash
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Creating directories..."
mkdir -p data/positive data/negative model

echo "Starting training..."
python train.py

echo "Generating template embedding..."
python generate_template.py

echo "Exporting ONNX model..."
python export_onnx.py

echo "Running inference demo..."
python infer_demo.py

echo "All done! ONNX model: model/hey_platin_wakeword.onnx"