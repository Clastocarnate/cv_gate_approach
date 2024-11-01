## Setup and Installation

1. Clone the repository:

   ```sh
   git clone https://github.com/your-username/cv_gate_approach.git
   cd cv_gate_approach
   ```

2. Install the necessary packages (requires OpenCV and NumPy):

   ```sh
   pip install opencv-python-headless numpy
   ```

## Usage

### Frame Extraction

Extract frames from `dehazed_footage.mp4`:

```sh
python get_frames.py
```

### Image Enhancement

Enhance an individual image to improve visibility:

```sh
python enhancement.py
```

### Smoothing Image

Apply Gaussian and median blurs to an enhanced image:

```sh
python blur.py
```

### Threshold Adjustment in YUV

Use the threshold tool to adjust YUV values interactively:

```sh
python threshold.py
```

### Full Integration and Processing

Run `integration.py` to apply the full processing pipeline, including image enhancement, blurring, and thresholding, with the option to navigate through images and save the output as a video:

```sh
python integration.py
```

## Key Functions

- **Image Enhancement**: Balances colors, improves contrast using CLAHE, and applies optional sharpening.
- **Smoothing**: Applies Gaussian and median blurs to reduce noise.
- **Threshold Adjustment**: Interactive adjustment of YUV thresholds for fine-tuning feature detection.
- **Integration and Video Saving**: Combines all processing steps and saves results as a video file for easier visualization.


  
