"""TV4 — đánh giá mô hình đã được chọn.

TODO:
1. Nạp checkpoint đã chọn bằng validation.
2. Đánh giá trên test set sau khi quá trình chọn model đã kết thúc.
3. Tính test loss, accuracy và các metric bổ sung đã thống nhất.
4. Trả kết quả có thể ghi vào outputs/experiment_summary.csv.
5. Không dùng test metrics để chọn hyperparameter/model.
"""
from typing import Any


def evaluate_model(model, test_loader, config: dict[str, Any]) -> dict[str, float]:
    """Trả về metrics đo thực tế sau khi triển khai."""
    raise NotImplementedError("TV4: triển khai theo docs/TEAM_CONTRACT.md")


def evaluate_model(model, test_loader, config: dict[str, Any]) -> dict[str, float]:
    """Đánh giá mô hình trên tập test và trả về các chỉ số."""

    import torch
    import torch.nn as nn
    import csv
    import os
    from sklearn.metrics import precision_score, recall_score, f1_score

    device = torch.device(
        config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )

    model = model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            predictions = torch.argmax(outputs, dim=1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_labels.extend(labels.cpu().numpy().tolist())
            all_predictions.extend(
                predictions.cpu().numpy().tolist()
            )

    if total_samples == 0:
        raise ValueError("Test loader không có dữ liệu.")

    test_loss = total_loss / total_samples

    correct = sum(
        true == pred
        for true, pred in zip(all_labels, all_predictions)
    )

    accuracy = correct / total_samples

    precision = precision_score(
        all_labels, all_predictions,
        average="weighted", zero_division=0
    )

    recall = recall_score(
        all_labels, all_predictions,
        average="weighted", zero_division=0
    )

    f1 = f1_score(
        all_labels, all_predictions,
        average="weighted", zero_division=0
    )

    results = {
        "test_loss": float(test_loss),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1)
    }

    print("\n===== KẾT QUẢ ĐÁNH GIÁ =====")

    for name, value in results.items():
        print(f"{name}: {value:.4f}")

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(
        output_dir, "experiment_summary.csv"
    )

    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results.keys())
        )

        if not file_exists or os.path.getsize(csv_path) == 0:
            writer.writeheader()

        writer.writerow(results)

    print(f"\nĐã lưu kết quả tại: {csv_path}")

    return results


    
