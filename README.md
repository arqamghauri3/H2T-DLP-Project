# Handwritten Text Recognition (HTR) with CRNN and CTC Loss

This project implements a Handwritten Text Recognition (HTR) system using a Convolutional Recurrent Neural Network (CRNN) with Connectionist Temporal Classification (CTC) loss. It processes grayscale images of handwritten words from the IAM Words dataset and transcribes them into text. The project is implemented in a Jupyter Notebook (H2TProject.ipynb) using TensorFlow and Keras.

## Table of Contents
Project Overview
Features
Dataset
Model Architecture

## Project Overview
The goal of this project is to build an HTR system that can accurately transcribe handwritten words from images. The system:
Preprocesses images to a uniform size (32x128 pixels, grayscale).
Encodes text labels into numerical sequences using a predefined character set.
Trains a CRNN model with CTC loss to align variable-length image sequences with text labels.
Performs inference and visualizes predictions on validation images.

## Features
CRNN Model: Combines convolutional layers for feature extraction and LSTM layers for sequence modeling.
CTC Loss: Handles unaligned sequence data, enabling transcription without pre-segmented characters.
IAM Words Dataset: Processes a standard dataset for HTR research.
Visualization: Displays predicted and ground truth text alongside images for qualitative evaluation.
Model Checkpointing: Saves the best model based on validation loss with a numerical naming scheme (e.g., models/1.keras).

## Dataset
The project uses the IAM Words dataset, which contains:
Images: Grayscale images of handwritten words (e.g., b06-110-07-03.png).
Metadata: A words.txt file with transcriptions and metadata (e.g., word ID, status, text).
Character Set: 78 characters (punctuation, digits, uppercase, and lowercase letters).

## Model Architecture
The CRNN model consists of:
Convolutional Layers: Extract features from (32, 128, 1) images.
Max Pooling: Reduces spatial dimensions.
LSTM Layers: Model temporal dependencies in the sequence.
Dense Layer: Outputs probabilities for 79 classes (78 characters + 1 blank token).
CTC Loss: Aligns predicted sequences with ground truth.
Input: (32, 128, 1) grayscale images. Output: (timesteps, 79) probabilities (e.g., 32 timesteps for the simplified model). Training Model: Includes CTC loss with additional inputs (labels, input_length, label_length).
