from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

@dataclass
class DatasetBundle:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    class_names: list[str]

LABEL_CANDIDATES = [
    "label", "Label", "class", "Class", "attack", "Attack", "type", "Type",
    "Attack_type", "attack_type", "category", "Category"
]


def _clean_frame(df: pd.DataFrame, label_column: str | None):
    df = df.replace([np.inf, -np.inf], np.nan).drop_duplicates()
    if label_column is None:
        label_column = next((c for c in LABEL_CANDIDATES if c in df.columns), None)
    if label_column is None or label_column not in df.columns:
        raise ValueError(f"Could not identify label column. Available columns: {list(df.columns)[:30]}")
    y = df[label_column].astype(str)
    x = df.drop(columns=[label_column])
    # Remove obvious IDs/timestamps with very high uniqueness if nonnumeric.
    drop_cols=[]
    for c in x.columns:
        if x[c].dtype == 'object' and x[c].nunique(dropna=True) > min(1000, 0.8*len(x)):
            drop_cols.append(c)
    if drop_cols:
        x=x.drop(columns=drop_cols)
    x = pd.get_dummies(x, dummy_na=True)
    x = x.apply(pd.to_numeric, errors='coerce')
    med = x.median(numeric_only=True)
    x = x.fillna(med).fillna(0.0)
    return x, y


def load_tabular_dataset(name: str, raw_dir: str, label_column: str | None = None,
                         max_rows: int | None = None, seed: int = 42,
                         synthetic_samples: int = 12000, synthetic_features: int = 30,
                         synthetic_classes: int = 5) -> DatasetBundle:
    if name.lower() == 'synthetic':
        x, y = make_classification(
            n_samples=synthetic_samples, n_features=synthetic_features,
            n_informative=max(5, synthetic_features//2), n_redundant=max(2, synthetic_features//6),
            n_classes=synthetic_classes, n_clusters_per_class=1,
            class_sep=1.5, random_state=seed
        )
        feature_names=[f"f{i}" for i in range(x.shape[1])]
        class_names=[str(i) for i in sorted(np.unique(y))]
    else:
        path=Path(raw_dir)/name
        csvs=sorted(path.rglob('*.csv')) if path.exists() else []
        if not csvs:
            raise FileNotFoundError(
                f"No CSV files found under {path}. Download/place the dataset there, "
                "or use dataset.name: synthetic for the built-in demo."
            )
        frames=[]
        remaining=max_rows
        for p in csvs:
            try:
                if remaining is not None and remaining <= 0: break
                nrows=remaining if remaining is not None else None
                d=pd.read_csv(p, low_memory=False, nrows=nrows)
                frames.append(d)
                if remaining is not None: remaining -= len(d)
            except Exception as e:
                print(f"Warning: skipping {p}: {e}")
        if not frames: raise RuntimeError("Dataset files were found but none could be loaded")
        df=pd.concat(frames, ignore_index=True)
        if max_rows and len(df)>max_rows:
            df=df.sample(max_rows, random_state=seed)
        xdf, yser=_clean_frame(df, label_column)
        feature_names=list(xdf.columns)
        le=LabelEncoder(); y=le.fit_transform(yser); class_names=list(le.classes_)
        x=xdf.to_numpy(dtype=np.float32)

    # 70/15/15 stratified split
    x_train, x_tmp, y_train, y_tmp = train_test_split(
        x, y, test_size=0.30, random_state=seed, stratify=y
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_tmp, y_tmp, test_size=0.50, random_state=seed, stratify=y_tmp
    )
    scaler=StandardScaler().fit(x_train)
    x_train=scaler.transform(x_train).astype(np.float32)
    x_val=scaler.transform(x_val).astype(np.float32)
    x_test=scaler.transform(x_test).astype(np.float32)
    return DatasetBundle(x_train,y_train.astype(np.int64),x_val,y_val.astype(np.int64),
                         x_test,y_test.astype(np.int64),feature_names,class_names)


def dirichlet_partition(y: np.ndarray, num_clients: int, alpha: float, seed: int,
                        min_size: int = 8) -> list[np.ndarray]:
    rng=np.random.default_rng(seed)
    classes=np.unique(y)
    for _ in range(100):
        parts=[[] for _ in range(num_clients)]
        for c in classes:
            idx=np.where(y==c)[0]; rng.shuffle(idx)
            proportions=rng.dirichlet(np.repeat(alpha,num_clients))
            cuts=(np.cumsum(proportions)*len(idx)).astype(int)[:-1]
            splits=np.split(idx,cuts)
            for i,s in enumerate(splits): parts[i].extend(s.tolist())
        sizes=[len(p) for p in parts]
        if min(sizes)>=min_size:
            return [np.asarray(rng.permutation(p),dtype=np.int64) for p in parts]
    # fallback balanced IID-ish split if extreme alpha/sample combination fails
    idx=rng.permutation(len(y)); return [np.asarray(a,dtype=np.int64) for a in np.array_split(idx,num_clients)]


def iid_partition(y: np.ndarray, num_clients: int, seed: int) -> list[np.ndarray]:
    rng=np.random.default_rng(seed); idx=rng.permutation(len(y))
    return [np.asarray(a,dtype=np.int64) for a in np.array_split(idx,num_clients)]
