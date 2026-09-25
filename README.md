# Adversarial Attack Analysis for Image Classification
A PyTorch-based tool for generating and analyzing adversarial examples against ImageNet image-classification models.  

The project implements FGSM, PGD, and Ensemble PGD attacks and evaluates their effects on multiple pretrained deep-learning models using both prediction-based and image-quality metrics.

## Overview

Adversarial examples are images that have been intentionally modified with small perturbations that can cause a neural network to change its prediction.

This project investigates how different adversarial attacks affect image classifiers while measuring the trade-off between:

* Attack effectiveness
* Perturbation magnitude
* Visual similarity to the original image
* Transferability across different models

The current implementation performs a detailed analysis on an individual input image.

## Models

The following pretrained ImageNet models are used:

* **ResNet-50**
* **Vision Transformer (ViT-B/16)**
* **EfficientNet-B0**

ResNet-50 is used as the primary model for FGSM and PGD attacks, while Ensemble PGD uses all three models simultaneously.

## Implemented Attacks

### FGSM

**Fast Gradient Sign Method (FGSM)** generates an adversarial perturbation using the sign of the gradient of the loss with respect to the input image.

### PGD

**Projected Gradient Descent (PGD)** performs multiple iterative gradient updates while constraining the perturbation within an \(L_\infty\) epsilon bound.

### Ensemble PGD

Ensemble PGD calculates the loss across multiple models and uses the combined gradient to generate an adversarial example.

This allows the project to examine whether adversarial perturbations transfer between different architectures.

## Experimental Settings

The default configuration evaluates three perturbation budgets:

```text
ε = 4/255
ε = 8/255
ε = 16/255
```

PGD-based attacks use:

```text
20 iterations
```

Input images are center-cropped to **224 × 224** before the attacks are generated.

## Evaluation Metrics

The project calculates several metrics to evaluate the generated adversarial examples.

### Prediction Change

The code measures whether each model's top-1 prediction changes after the attack.

The current implementation reports this as:

```text
Attack Success Rate (%)
```

More precisely, this represents the **percentage of evaluated models whose top-1 prediction changed**, rather than the conventional ground-truth-based attack success rate.

### Confidence Drop

The difference between the original and protected image's top-1 prediction confidence is calculated for each model.

### MSE

**Mean Squared Error (MSE)** measures the average pixel-level difference between the original and adversarial images.

Lower values indicate smaller pixel-level differences.

### PSNR

**Peak Signal-to-Noise Ratio (PSNR)** is used to evaluate image similarity.

Higher PSNR generally indicates that the adversarial image is closer to the original image.

### SSIM

**Structural Similarity Index (SSIM)** measures structural similarity between the original and adversarial images.

Values closer to 1 indicate greater structural similarity.

### Perturbation Statistics

The project also reports:

* L2 norm
* Mean perturbation
* Standard deviation of perturbation
* Maximum perturbation
* Percentage of pixels with perturbation greater than 10 intensity levels

## Dataset

The experiments were conducted using a subset of the **ILSVRC2012 (ImageNet) validation dataset**. A total of **200 validation images** were used for the dataset-level adversarial attack analysis. As well as some hand-drawn images of an anonymous artist. 

The images were evaluated using FGSM, PGD, and Ensemble PGD across multiple perturbation levels (ε = 4/255, 8/255, and 16/255).


## Workflow

```text
Input Image
     │
     ▼
Resize + Center Crop
     │
     ▼
224 × 224 Image
     │
     ├──────────────┐
     ▼              ▼
Original        Adversarial
Prediction       Attacks
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
        FGSM       PGD    Ensemble PGD
          │         │         │
          └─────────┼─────────┘
                    ▼
          Evaluate Multiple Models
                    │
                    ▼
       ┌─────────────────────────┐
       │ PSNR / SSIM / MSE       │
       │ Prediction Changes      │
       │ Confidence Changes      │
       │ Perturbation Statistics │
       └─────────────────────────┘
```

## Output

For each analyzed image and epsilon value, the program generates:

* Cropped original image
* Adversarial image
* Full-size resized adversarial image
* Comparison visualizations
* Perturbation heatmaps
* Analysis report
* CSV results
* Epsilon comparison plots

The generated files are organized into separate output directories.

## Code Overview
There are two code files in this project. You can use them depending on your goal. They use the same core attack methodology, but the main distinction is how many images they test and how the results are analyzed. 

### `codes.py`

Single-image adversarial attack analysis.

This script takes **one image at a time** and generates adversarial examples using:

* FGSM
* PGD
* Ensemble PGD

The generated images are evaluated using **ResNet-50, ViT-B/16, and EfficientNet-B0**. It compares the original and adversarial images using prediction changes, Attack Success Rate (ASR), confidence changes, PSNR, SSIM, MSE, and perturbation metrics.

It also generates visual comparisons and perturbation heatmaps for detailed inspection.

### `smallcopycat.py`

Dataset-level adversarial attack experiment.

This script applies the same three attack methods to **multiple images (200+ or more)** and aggregates the results across the dataset.

It is used to compare FGSM, PGD, and Ensemble PGD in terms of:

* Attack Success Rate
* Prediction changes across models
* Confidence changes
* Image quality
* Perturbation magnitude
* Performance across different epsilon values

The script outputs per-image metrics, dataset-level summaries, CSV files, reports, and visualizations.

### Purpose

The two scripts provide two levels of analysis:

**`codes.py` → detailed analysis of individual adversarial examples**

**`smallcopycat.py` → quantitative evaluation across a larger dataset**



## Example Results

If you extract **`analysis_outputs.zip`**, you can view some of the results generated by **`codes.py`**. These results show the analysis performed on a single image. The original image of these results is 'blue_man.png.

The results generated by **`smallcopycat.py`** were too large to upload in full because the script analyzed 200 images and saved detailed results for 20 example images. However, four representative results are included below: a single-image comparison of the different attack methods at ε = 8/255, a summary comparing attack success and visual quality, an attack success count across the tested images, and the dataset-level adversarial comparison report for all 200 images.

![Adversarial Attack Comparison](https://raw.githubusercontent.com/kagylain/Adversarial-Attack-Analysis-/5cc55ead8c9760c14f859ece199dcc9fd72d4a02/dataset_3eps_20iters_example18_eps8_comparison.png)


![Quality vs Attack Success](https://raw.githubusercontent.com/kagylain/Adversarial-Attack-Analysis-/c6307a6df64fbda96b4c95c51693b83a51b6697d/dataset_3eps_20iters_quality_vs_success.png)

![Attack Success Count](https://raw.githubusercontent.com/kagylain/Adversarial-Attack-Analysis-/c6307a6df64fbda96b4c95c51693b83a51b6697d/dataset_3eps_20iters_success_counts.png)

## Analysis & Documentation

The repository also includes explanatory PDF slides documenting the project workflow, methodology, experimental setup, and results.

Additional analysis was performed using **Raphael AI** to examine the visual changes and behavior of images after applying the different adversarial attack methods. The corresponding observations and examples are presented in the included PDF slides. The image used for this experiment is 'girlsitting.jpg'. 

A representative example of this analysis is shown below:

![Raphael AI Analysis](https://raw.githubusercontent.com/kagylain/Adversarial-Attack-Analysis-/4eeb1258e3a1a5f76d3a8b5cfa2c9084ba26efed/Rapahel%20AI%20.png)



## Limitations

This project includes both **single-image analysis** and **dataset-level evaluation** using a 200-image subset of the ILSVRC2012 validation dataset.

Important limitations include:

* The dataset-level experiment is limited to 200 images and may not represent the full diversity of the ImageNet validation dataset.
* No manually provided ground-truth labels are currently used in the evaluation. Therefore, attack success is primarily assessed through changes in model predictions rather than whether the model's prediction changes from a correct label to an incorrect one.
* The reported prediction-change rate measures changes in top-1 predictions across the evaluated models.
* FGSM and PGD are optimized using the ResNet-50 gradient, while Ensemble PGD combines gradients from ResNet-50, ViT-B/16, and EfficientNet-B0.
* The adversarial attacks are generated at 224 × 224 resolution.
* The "full-size" adversarial images are produced by resizing the 224 × 224 adversarial images back to their original image dimensions.
* PGD uses random initialization, so results may vary between runs unless a random seed is fixed.
* The experiments focus on untargeted adversarial attacks.
* The analysis uses pretrained ImageNet models, so the results reflect the behavior of these specific architectures and weights.

## Future Improvements

Potential extensions include:

* Increase the number and diversity of images used in the dataset evaluation.
* Incorporate ground-truth labels to calculate conventional attack success rates.
* Report transferability separately for each target model.
* Add reproducible random seeds.
* Freeze model parameters during attack generation.
* Further optimize Apple Silicon MPS support.
* Compare additional adversarial attack methods.
* Evaluate targeted attacks.
* Perform more extensive statistical analysis across the dataset.
* Compare the computational cost and runtime of different attack methods.
* Investigate the relationship between perturbation magnitude, image quality, and attack success.
* Evaluate the attacks on additional datasets and model architectures.

## Technologies

* Python
* PyTorch
* Torchvision
* NumPy
* Pillow
* Matplotlib
* scikit-image


## Disclaimer

This project is intended for research and educational purposes, particularly for studying the robustness of machine-learning image classifiers against adversarial perturbations.
