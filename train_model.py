"""
train_model.py
--------------
เทรนโมเดล CNN สำหรับรู้จำตัวเลขลายมือ (MNIST) จากไฟล์ CSV
รูปแบบ CSV: คอลัมน์แรก = label (0-9), 784 คอลัมน์ถัดไป = พิกเซล 28×28 (0-255)

ตัวอย่างการใช้งาน
    python train_model.py
    python train_model.py --csv Train.csv --output models/mnist_model.keras --epochs 8
"""

import argparse
import os
import sys

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# คอนโซล Windows บางเครื่องใช้ codepage ที่พิมพ์อักขระบางตัวไม่ได้
# ตั้ง errors="replace" เพื่อไม่ให้สคริปต์ crash ระหว่างเทรน
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:
        pass

import numpy as np
import pandas as pd
from tensorflow.keras import layers, models

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(BASE_DIR, "Train.csv")
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "models", "mnist_model.keras")

VALIDATION_RATIO = 0.1
RANDOM_SEED = 42


def load_training_data(csv_path):
    print(f"กำลังโหลดข้อมูลจาก {csv_path} ...")
    training_data = pd.read_csv(csv_path)

    if training_data.shape[1] != 785:
        raise ValueError(
            f"Expected 785 columns (label + 784 pixels), found {training_data.shape[1]}."
        )

    labels = training_data.iloc[:, 0].to_numpy(dtype=np.int64)
    pixels = training_data.iloc[:, 1:].to_numpy(dtype=np.float32)
    images = pixels.reshape(-1, 28, 28, 1) / 255.0

    if not np.isin(labels, np.arange(10)).all():
        raise ValueError("Labels must contain only digits from 0 through 9.")

    print(f"โหลดสำเร็จ: {images.shape[0]} ภาพ (28x28x1)")
    return images, labels


def create_model():
    """สถาปัตยกรรม CNN (เหมือนกับโมเดลที่โปรเจกต์นี้ใช้งาน)"""
    return models.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(128, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ]
    )


def _print_per_class_report(labels_true, labels_pred):
    print("\n--- ความแม่นยำรายคลาส (ชุด validation) ---")
    for digit in range(10):
        mask = labels_true == digit
        if mask.sum() == 0:
            continue
        acc = (labels_pred[mask] == digit).mean() * 100
        bar = "█" * int(round(acc / 5))
        print(f"  ตัวเลข {digit}: {acc:6.2f}%  {bar}")


def train(csv_path, output_path, epochs, batch_size=64):
    images, labels = load_training_data(csv_path)
    model = create_model()
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # แบ่งชุด validation แบบกำหนด seed ได้ (ทำซ้ำได้ผลเดิม)
    rng = np.random.default_rng(RANDOM_SEED)
    order = rng.permutation(len(labels))
    n_val = max(1, int(len(labels) * VALIDATION_RATIO))
    val_idx, train_idx = order[:n_val], order[n_val:]

    print(
        f"\nเริ่มเทรน {epochs} epochs · "
        f"train {len(train_idx)} / val {len(val_idx)} ภาพ"
    )
    history = model.fit(
        images[train_idx],
        labels[train_idx],
        validation_data=(images[val_idx], labels[val_idx]),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
    )

    # สรุปผลลัพธ์
    val_loss, val_acc = model.evaluate(
        images[val_idx], labels[val_idx], verbose=0
    )
    predictions = model.predict(images[val_idx], verbose=0).argmax(axis=1)
    _print_per_class_report(labels[val_idx], predictions)

    best_acc = max(history.history.get("val_accuracy", [val_acc]))
    print(f"\nความแม่นยำชุด validation: {val_acc * 100:.2f}% (สูงสุดระหว่างเทรน {best_acc * 100:.2f}%)")

    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    model.save(output_path)
    print(f"บันทึกโมเดลที่: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train an MNIST digit recognition model."
    )
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Training CSV path")
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT, help="Output model path (.keras)"
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    args = parser.parse_args()

    train(args.csv, args.output, args.epochs, args.batch_size)
