# Week 7 — EEG Time-Series Classification Pipeline

## Overview
This pipeline classifies EEG windows as eye-open or eye-closed using the 
UCI/OpenML EEG Eye State dataset (14 channels, 14,980 time samples, single subject).
It demonstrates correct time-series handling: sliding-window segmentation, 
per-window feature extraction, leakage-safe splitting, and error analysis.

## Pipeline steps
1. **Load data** (`load_eeg_data`): reads the CSV, separates 14 EEG channels 
   from the binary `class` label.
2. **Segment** (`create_sliding_windows`): splits the continuous signal into 
   128-sample windows with a 64-sample step (50% overlap). Produces 233 windows.
3. **Extract features** (`extract_features`): computes mean and std per 
   channel per window -> 28 features per window.
4. **Split** (`time_block_split`): splits windows by TIME ORDER (not randomly), 
   with a buffer to remove any remaining overlap leakage at the boundary.
5. **Train & evaluate** (`train_and_evaluate`): fits any sklearn-style 
   classifier, returns train/test accuracy and confusion matrix.

## How to run