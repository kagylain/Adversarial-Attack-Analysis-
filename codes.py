import os
import csv
import logging
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, vit_b_16, efficientnet_b0
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights, EfficientNet_B0_Weights

import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')


class AdversarialAttackAnalyzer:
    """
    FGSM / PGD / Ensemble PGD adversarial image protection analyzer.

    IMPORTANT FIX IN THIS VERSION:
    - All attacks are generated in 0-1 pixel space, not normalized ImageNet space.
    - Quality metrics are calculated between images of the same size: 224x224 cropped original
      vs 224x224 cropped protected image.
    - The code saves:
        1) cropped original image used by models
        2) cropped protected images, 224x224
        3) full-size protected images, resized back to the original input dimensions
        4) comparison figures, heatmaps, report, and CSV metrics
    """

    def __init__(self, use_gpu=True, output_dir="analysis_outputs"):
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.cropped_original_dir = self.output_dir / "cropped_original"
        self.cropped_protected_dir = self.output_dir / "cropped_protected_224"
        self.fullsize_protected_dir = self.output_dir / "protected_full_size"
        self.figures_dir = self.output_dir / "figures"
        self.reports_dir = self.output_dir / "reports"

        for folder in [
            self.cropped_original_dir,
            self.cropped_protected_dir,
            self.fullsize_protected_dir,
            self.figures_dir,
            self.reports_dir,
        ]:
            folder.mkdir(exist_ok=True)

        logging.info("Loading models...")
        self.models = {
            "resnet50": resnet50(weights=ResNet50_Weights.DEFAULT),
            "vit": vit_b_16(weights=ViT_B_16_Weights.DEFAULT),
            "efficientnet": efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT),
        }

        self.class_names = ResNet50_Weights.DEFAULT.meta["categories"]

        for model in self.models.values():
            model.to(self.device)
            model.eval()

        # Used to create the exact 224x224 image that all models see.
        self.crop_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ])

        # Converts PIL image to 0-1 tensor without ImageNet normalization.
        self.to_tensor = transforms.ToTensor()

        # ImageNet normalization is applied only inside the model forward pass.
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    # ------------------------------------------------------------------
    # Main analysis function
    # ------------------------------------------------------------------
    def analyze_and_compare(
        self,
        image_path,
        epsilon_values=(4/255, 8/255, 16/255),
        iterations=20,
        output_prefix="comparison",
    ):
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        img_pil_original = Image.open(image_path).convert("RGB")
        original_width, original_height = img_pil_original.size

        # This is the exact image size used by ResNet50, ViT, and EfficientNet.
        img_pil_cropped = self.crop_transform(img_pil_original)
        cropped_original_np = np.array(img_pil_cropped)
        cropped_original_path = self.cropped_original_dir / f"{output_prefix}_cropped_original_224.png"
        img_pil_cropped.save(cropped_original_path)
        logging.info(f"Saved cropped original: {cropped_original_path}")

        original_predictions = self._get_model_predictions(cropped_original_np)
        logging.info("\nOriginal cropped image predictions:")
        for model_name, pred in original_predictions.items():
            logging.info(f"  {model_name}: {pred['label']} ({pred['confidence']:.2f}%)")

        results = {}

        for eps in epsilon_values:
            eps_label = f"{int(round(eps * 255))}"
            logging.info(f"\n{'=' * 60}")
            logging.info(f"Testing epsilon = {eps:.6f} ({eps_label}/255)")
            logging.info(f"{'=' * 60}")

            attacks = {
                "FGSM": self._fgsm_attack(cropped_original_np, eps),
                "PGD": self._pgd_attack(cropped_original_np, eps, iterations),
                "Ensemble PGD": self._ensemble_pgd_attack(cropped_original_np, eps, iterations),
            }

            metrics = {}
            protected_predictions = {}

            for method_name, protected_224_np in attacks.items():
                safe_method = self._safe_name(method_name)

                # Save cropped protected image exactly as the model sees it.
                cropped_protected_path = (
                    self.cropped_protected_dir
                    / f"{output_prefix}_eps{eps_label}_{safe_method}_protected_224.png"
                )
                Image.fromarray(protected_224_np).save(cropped_protected_path)
                logging.info(f"Saved cropped protected: {cropped_protected_path}")

                # Save full-size protected image for website / visual testing.
                # This is the 224x224 protected result resized back to the original image dimensions.
                fullsize_img = Image.fromarray(protected_224_np).resize(
                    (original_width, original_height),
                    Image.Resampling.BICUBIC,
                )
                fullsize_path = (
                    self.fullsize_protected_dir
                    / f"{output_prefix}_eps{eps_label}_{safe_method}_protected_fullsize.png"
                )
                fullsize_img.save(fullsize_path)
                logging.info(f"Saved full-size protected: {fullsize_path}")

                # IMPORTANT: metrics compare cropped original 224x224 vs protected 224x224.
                metrics[method_name] = self._calculate_all_metrics(cropped_original_np, protected_224_np)

                protected_predictions[method_name] = self._get_model_predictions(protected_224_np)
                effectiveness = self._calculate_prediction_effectiveness(
                    original_predictions,
                    protected_predictions[method_name],
                )
                metrics[method_name].update(effectiveness)

                logging.info(f"\n{method_name} Metrics:")
                for metric_name, value in metrics[method_name].items():
                    logging.info(f"  {metric_name}: {value}")

            results[eps_label] = {
                "attacks": attacks,
                "metrics": metrics,
                "original_predictions": original_predictions,
                "protected_predictions": protected_predictions,
            }

            self._visualize_comparison(
                cropped_original_np,
                attacks,
                metrics,
                eps_label,
                self.figures_dir / f"{output_prefix}_eps{eps_label}_comparison.png",
            )

            self._visualize_perturbations(
                cropped_original_np,
                attacks,
                eps_label,
                self.figures_dir / f"{output_prefix}_eps{eps_label}_perturbations.png",
            )

        report_path = self.reports_dir / f"{output_prefix}_report.txt"
        csv_path = self.reports_dir / f"{output_prefix}_metrics.csv"
        self._generate_report(results, report_path)
        self._export_metrics_csv(results, csv_path)

        if len(epsilon_values) > 1:
            self._plot_epsilon_comparison(
                results,
                self.figures_dir / f"{output_prefix}_epsilon_comparison.png",
            )

        return results

    # ------------------------------------------------------------------
    # Attack methods
    # ------------------------------------------------------------------
    def _fgsm_attack(self, image_224_np, epsilon):
        """Fast Gradient Sign Method in 0-1 pixel space."""
        img_tensor = self._prepare_pixel_tensor(image_224_np)
        img_tensor.requires_grad = True

        output = self._forward_model("resnet50", img_tensor)
        top_class = output.argmax(dim=1)
        loss = nn.functional.cross_entropy(output, top_class)
        loss.backward()

        perturbation = epsilon * img_tensor.grad.sign()
        perturbed = torch.clamp(img_tensor + perturbation, 0.0, 1.0)

        return self._pixel_tensor_to_image(perturbed)

    def _pgd_attack(self, image_224_np, epsilon, iterations=10):
        """Projected Gradient Descent in 0-1 pixel space."""
        img_tensor = self._prepare_pixel_tensor(image_224_np)
        alpha = epsilon / max(iterations, 1) * 2

        perturbed = img_tensor + torch.empty_like(img_tensor).uniform_(-epsilon, epsilon)
        perturbed = torch.clamp(perturbed, 0.0, 1.0)

        for _ in range(iterations):
            perturbed = perturbed.detach()
            perturbed.requires_grad = True

            output = self._forward_model("resnet50", perturbed)
            top_class = output.argmax(dim=1)
            loss = nn.functional.cross_entropy(output, top_class)
            loss.backward()

            with torch.no_grad():
                perturbed = perturbed + alpha * perturbed.grad.sign()
                delta = torch.clamp(perturbed - img_tensor, -epsilon, epsilon)
                perturbed = torch.clamp(img_tensor + delta, 0.0, 1.0)

        return self._pixel_tensor_to_image(perturbed)

    def _ensemble_pgd_attack(self, image_224_np, epsilon, iterations=10):
        """Ensemble PGD using ResNet50, ViT, and EfficientNet in 0-1 pixel space."""
        img_tensor = self._prepare_pixel_tensor(image_224_np)
        alpha = epsilon / max(iterations, 1) * 2

        perturbed = img_tensor + torch.empty_like(img_tensor).uniform_(-epsilon, epsilon)
        perturbed = torch.clamp(perturbed, 0.0, 1.0)

        for _ in range(iterations):
            perturbed = perturbed.detach()
            perturbed.requires_grad = True

            total_loss = 0
            for model_name in self.models.keys():
                output = self._forward_model(model_name, perturbed)
                top_class = output.argmax(dim=1)
                total_loss = total_loss + nn.functional.cross_entropy(output, top_class)

            total_loss.backward()

            with torch.no_grad():
                perturbed = perturbed + alpha * perturbed.grad.sign()
                delta = torch.clamp(perturbed - img_tensor, -epsilon, epsilon)
                perturbed = torch.clamp(img_tensor + delta, 0.0, 1.0)

        return self._pixel_tensor_to_image(perturbed)

    # ------------------------------------------------------------------
    # Prediction and metrics
    # ------------------------------------------------------------------
    def _get_model_predictions(self, image_224_np):
        """Return top-1 ImageNet prediction and confidence for each model."""
        img_tensor = self._prepare_pixel_tensor(image_224_np)
        predictions = {}

        with torch.no_grad():
            for model_name in self.models.keys():
                output = self._forward_model(model_name, img_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, class_idx = torch.max(probabilities, dim=1)
                class_idx = int(class_idx.item())
                confidence = float(confidence.item() * 100)

                predictions[model_name] = {
                    "class_index": class_idx,
                    "label": self.class_names[class_idx],
                    "confidence": confidence,
                }

        return predictions

    def _calculate_prediction_effectiveness(self, original_predictions, protected_predictions):
        """Calculate prediction change, ASR, and average confidence drop."""
        changed_count = 0
        confidence_drops = []
        details = []

        for model_name in self.models.keys():
            original = original_predictions[model_name]
            protected = protected_predictions[model_name]

            prediction_changed = original["class_index"] != protected["class_index"]
            if prediction_changed:
                changed_count += 1

            confidence_drop = original["confidence"] - protected["confidence"]
            confidence_drops.append(confidence_drop)

            details.append(
                f"{model_name}: {original['label']} ({original['confidence']:.2f}%) -> "
                f"{protected['label']} ({protected['confidence']:.2f}%)"
            )

        total_models = len(self.models)
        attack_success_rate = changed_count / total_models * 100
        avg_confidence_drop = float(np.mean(confidence_drops))

        return {
            "Prediction Changes": f"{changed_count}/{total_models}",
            "Attack Success Rate (%)": f"{attack_success_rate:.2f}",
            "Avg Confidence Drop (%)": f"{avg_confidence_drop:.2f}",
            "Prediction Details": " | ".join(details),
        }

    def _calculate_all_metrics(self, original_224_np, protected_224_np):
        """Calculate quality metrics on same-sized 224x224 images."""
        original = original_224_np.astype(np.float32)
        protected = protected_224_np.astype(np.float32)

        mse = np.mean((original - protected) ** 2)
        perturbation = np.abs(original - protected)

        if mse == 0:
            psnr_value = float("inf")
        else:
            psnr_value = 20 * np.log10(255.0 / np.sqrt(mse))

        ssim_value = ssim(
            original_224_np,
            protected_224_np,
            channel_axis=2,
            data_range=255,
        )

        l2_norm = np.linalg.norm(perturbation)
        significant_changes = np.sum(perturbation > 10) / perturbation.size * 100

        return {
            "PSNR (dB)": f"{psnr_value:.2f}",
            "SSIM": f"{ssim_value:.4f}",
            "MSE": f"{mse:.2f}",
            "Max Perturbation": f"{perturbation.max():.2f}",
            "Mean Perturbation": f"{perturbation.mean():.2f}",
            "Std Perturbation": f"{perturbation.std():.2f}",
            "L2 Norm": f"{l2_norm:.2f}",
            "Pixels Changed >10 (%)": f"{significant_changes:.2f}",
        }

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    def _visualize_comparison(self, original_224_np, attacks, metrics, eps_label, output_path):
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f"Attack Method Comparison on Cropped 224x224 Image (ε = {eps_label}/255)", fontsize=16)

        axes[0, 0].imshow(original_224_np)
        axes[0, 0].set_title("Original Cropped 224x224")
        axes[0, 0].axis("off")

        for idx, (name, img) in enumerate(attacks.items(), 1):
            axes[0, idx].imshow(img)
            axes[0, idx].set_title(name)
            axes[0, idx].axis("off")

        axes[1, 0].text(
            0.5,
            0.5,
            "Perturbation Maps\n(5x amplified)",
            ha="center",
            va="center",
            fontsize=12,
        )
        axes[1, 0].axis("off")

        for idx, (name, img) in enumerate(attacks.items(), 1):
            diff = np.abs(original_224_np.astype(float) - img.astype(float)) * 5
            diff = np.clip(diff, 0, 255).astype(np.uint8)
            axes[1, idx].imshow(diff)

            metric_text = (
                f"PSNR: {metrics[name]['PSNR (dB)']}\n"
                f"SSIM: {metrics[name]['SSIM']}\n"
                f"Mean Pert: {metrics[name]['Mean Perturbation']}\n"
                f"ASR: {metrics[name]['Attack Success Rate (%)']}%"
            )
            axes[1, idx].text(
                0.02,
                0.98,
                metric_text,
                transform=axes[1, idx].transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )
            axes[1, idx].axis("off")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logging.info(f"Saved comparison: {output_path}")

    def _visualize_perturbations(self, original_224_np, attacks, eps_label, output_path):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Perturbation Heatmaps on Cropped 224x224 Image (ε = {eps_label}/255)", fontsize=14)

        for idx, (name, img) in enumerate(attacks.items()):
            perturbation = np.abs(original_224_np.astype(float) - img.astype(float))
            perturbation_magnitude = np.mean(perturbation, axis=2)

            im = axes[idx].imshow(perturbation_magnitude, cmap="hot", interpolation="nearest")
            axes[idx].set_title(f"{name}\nMean: {perturbation.mean():.2f}")
            axes[idx].axis("off")
            plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logging.info(f"Saved perturbation heatmaps: {output_path}")

    def _plot_epsilon_comparison(self, results, output_path):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Attack Method Performance Across Epsilon Values", fontsize=16)

        eps_values = list(results.keys())
        methods = ["FGSM", "PGD", "Ensemble PGD"]
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]

        for method, color in zip(methods, colors):
            values = [float(results[eps]["metrics"][method]["PSNR (dB)"]) for eps in eps_values]
            axes[0, 0].plot(eps_values, values, marker="o", label=method, color=color)
        axes[0, 0].set_xlabel("Epsilon (x/255)")
        axes[0, 0].set_ylabel("PSNR (dB)")
        axes[0, 0].set_title("Image Quality: PSNR")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        for method, color in zip(methods, colors):
            values = [float(results[eps]["metrics"][method]["SSIM"]) for eps in eps_values]
            axes[0, 1].plot(eps_values, values, marker="s", label=method, color=color)
        axes[0, 1].set_xlabel("Epsilon (x/255)")
        axes[0, 1].set_ylabel("SSIM")
        axes[0, 1].set_title("Structural Similarity: SSIM")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        for method, color in zip(methods, colors):
            values = [float(results[eps]["metrics"][method]["Mean Perturbation"]) for eps in eps_values]
            axes[1, 0].plot(eps_values, values, marker="^", label=method, color=color)
        axes[1, 0].set_xlabel("Epsilon (x/255)")
        axes[1, 0].set_ylabel("Mean Perturbation")
        axes[1, 0].set_title("Average Pixel Change")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        for method, color in zip(methods, colors):
            values = [float(results[eps]["metrics"][method]["Attack Success Rate (%)"]) for eps in eps_values]
            axes[1, 1].plot(eps_values, values, marker="d", label=method, color=color)
        axes[1, 1].set_xlabel("Epsilon (x/255)")
        axes[1, 1].set_ylabel("ASR (%)")
        axes[1, 1].set_title("Attack Success Rate")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logging.info(f"Saved epsilon comparison: {output_path}")

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------
    def _generate_report(self, results, output_path):
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("ADVERSARIAL ATTACK COMPARISON REPORT\n")
            f.write("Metrics are calculated on cropped 224x224 images.\n")
            f.write("=" * 70 + "\n\n")

            f.write("METRIC INTERPRETATION:\n")
            f.write("-" * 70 + "\n")
            f.write("PSNR: higher is better. 30-40 dB usually indicates good visual quality.\n")
            f.write("SSIM: closer to 1 means more structurally similar.\n")
            f.write("MSE: lower means less pixel-level distortion.\n")
            f.write("Mean Perturbation: average pixel change on 0-255 scale.\n")
            f.write("ASR: percentage of local ImageNet models whose top prediction changed.\n")
            f.write("Avg Confidence Drop: positive means average confidence decreased.\n")
            f.write("\n")

            for eps_label, data in results.items():
                f.write(f"\nEPSILON = {eps_label}/255\n")
                f.write("=" * 70 + "\n\n")

                for method, metrics in data["metrics"].items():
                    f.write(f"{method}:\n")
                    for metric_name, value in metrics.items():
                        f.write(f"  {metric_name:.<30} {value}\n")
                    f.write("\n")

                f.write("-" * 70 + "\n")

        logging.info(f"Saved report: {output_path}")

    def _export_metrics_csv(self, results, output_path):
        rows = []
        for eps_label, data in results.items():
            for method, metrics in data["metrics"].items():
                row = {"epsilon": f"{eps_label}/255", "method": method}
                row.update(metrics)
                rows.append(row)

        if not rows:
            return

        fieldnames = list(rows[0].keys())
        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        logging.info(f"Saved metrics CSV: {output_path}")

    # ------------------------------------------------------------------
    # Tensor helpers
    # ------------------------------------------------------------------
    def _prepare_pixel_tensor(self, image_224_np):
        """Convert 224x224 RGB numpy image to 0-1 tensor."""
        img_pil = Image.fromarray(image_224_np.astype(np.uint8))
        return self.to_tensor(img_pil).unsqueeze(0).to(self.device)

    def _forward_model(self, model_name, pixel_tensor):
        """Normalize 0-1 tensor before sending it to ImageNet model."""
        normalized = self.normalize(pixel_tensor.squeeze(0)).unsqueeze(0)
        return self.models[model_name](normalized)

    def _pixel_tensor_to_image(self, pixel_tensor):
        """Convert 0-1 tensor back to 224x224 RGB uint8 image."""
        arr = pixel_tensor.detach().cpu().squeeze(0).numpy().transpose(1, 2, 0)
        arr = np.clip(arr * 255.0, 0, 255).round().astype(np.uint8)
        return arr

    @staticmethod
    def _safe_name(name):
        return name.lower().replace(" ", "_").replace("/", "_")


if __name__ == "__main__":
    analyzer = AdversarialAttackAnalyzer(use_gpu=True, output_dir="analysis_outputs")

    results = analyzer.analyze_and_compare(
        image_path="whitegirl.jpg",
        epsilon_values=[4/255, 8/255, 16/255],
        iterations=20,
        output_prefix="whitegirl_analysis",
    )

    print("\n" + "=" * 70)
    print("Analysis complete! Generated folders:")
    print("  - analysis_outputs/cropped_original")
    print("  - analysis_outputs/cropped_protected_224")
    print("  - analysis_outputs/protected_full_size")
    print("  - analysis_outputs/figures")
    print("  - analysis_outputs/reports")
    print("=" * 70)
