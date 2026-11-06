# Train SIFT + Bag-of-Visual-Words classifiers on the Intel Image Classification dataset.
# Usage:  python train.py --data intel_data --out models
# (--data is the folder that contains seg_train and seg_test, anywhere inside it)

import argparse
import os
import random

import cv2
import joblib
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

IMG_SIZE = 128
K = 100


def find_split(root, name):
    for folder, subs, files in os.walk(root):
        if os.path.basename(folder) == name and "buildings" in subs:
            return folder
    raise FileNotFoundError(name)


def load_split(folder, classes, sift):
    # returns SIFT descriptors and labels for every image in the folder
    all_desc, labels = [], []
    for idx, name in enumerate(classes):
        class_folder = os.path.join(folder, name)
        for file in os.listdir(class_folder):
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img = cv2.imread(os.path.join(class_folder, file))
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            desc = sift.detectAndCompute(gray, None)[1]
            all_desc.append(desc)
            labels.append(idx)
    return all_desc, np.array(labels)


def to_histogram(desc, kmeans):
    hist = np.zeros(K)
    if desc is not None:
        for w in kmeans.predict(desc.astype(np.float32)):
            hist[w] += 1
        hist = hist / hist.sum()
    return hist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="folder with seg_train and seg_test")
    parser.add_argument("--out", default="models", help="folder to save the model")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    sift = cv2.SIFT_create()

    train_dir = find_split(args.data, "seg_train")
    test_dir = find_split(args.data, "seg_test")
    classes = sorted(os.listdir(train_dir))
    print("Classes:", classes)

    print("Extracting SIFT features...")
    train_desc, y_train = load_split(train_dir, classes, sift)
    test_desc, y_test = load_split(test_dir, classes, sift)

    print("Building visual vocabulary...")
    stacked = np.vstack([d for d in train_desc if d is not None])
    if len(stacked) > 200000:
        stacked = stacked[np.random.choice(len(stacked), 200000, replace=False)]
    kmeans = MiniBatchKMeans(n_clusters=K, random_state=42, batch_size=2048, n_init=3)
    kmeans.fit(stacked)

    X_train = np.array([to_histogram(d, kmeans) for d in train_desc])
    X_test = np.array([to_histogram(d, kmeans) for d in test_desc])

    models = {
        "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, probability=True, random_state=42)),
        "Random Forest": make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
        "k-NN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        results[name] = accuracy_score(y_test, model.predict(X_test))
        print(f"{name}: test accuracy = {results[name]:.4f}")

    best_name = max(results, key=results.get)
    print("Best model:", best_name)

    os.makedirs(args.out, exist_ok=True)
    bundle = {
        "model": models[best_name],
        "kmeans": kmeans,
        "classes": classes,
        "img_size": IMG_SIZE,
        "k": K,
        "model_name": best_name,
        "test_accuracy": results[best_name],
    }
    joblib.dump(bundle, os.path.join(args.out, "best_model.joblib"), compress=3)
    print("Saved", os.path.join(args.out, "best_model.joblib"))


if __name__ == "__main__":
    main()
