

"""
model_pipeline.py
=================

Production-style modeling pipeline for India Credit Risk Intelligence.

The pipeline deliberately answers several modeling questions:

1. Why classification?
   The available label is binary high-risk vs low-risk.

2. Why not accuracy alone?
   Credit-risk decisions are asymmetric. False negatives can be costly,
   therefore precision, recall, F1, ROC-AUC and PR-AUC are reported.

3. Why XGBoost?
   It captures nonlinear interactions and generally performs strongly
   on heterogeneous tabular credit data.

4. Why Logistic Regression?
   It is used as an interpretable baseline. We do not claim XGBoost
   is superior until the measured results demonstrate it.

5. How is leakage prevented?
   Target-derived fields are explicitly excluded.

6. What about CIBIL / Credit_Score?
   Credit_Score and its derived cibil_band are retained in the Silver
   analytical dataset for borrower-level analysis and downstream
   reporting, but they are excluded from the production model.
   Credit_Score is excluded because the target itself is a risk
   classification constructed from credit-risk signals. Including it
   would risk learning the label-generation mechanism rather than an
   independently useful predictive relationship. cibil_band is excluded
   for the same reason because it is derived directly from Credit_Score.

7. How is class imbalance handled?
   scale_pos_weight is calculated from the training data only.

8. How is the decision threshold chosen?
   A threshold analysis is produced instead of blindly assuming 0.50.

9. How is SHAP used?
   SHAP explains the trained tree model's feature contributions.

Run through:
    python -m src.analytics.run_ml_model
"""

from __future__ import annotations

import json
import pickle
import warnings

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")


try:
    import xgboost as xgb
except ImportError as exc:
    raise ImportError(
        "xgboost is required. Install with: pip install xgboost"
    ) from exc


# Optional SHAP.
try:
    import shap

    SHAP_AVAILABLE = True

except ImportError:
    SHAP_AVAILABLE = False


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SILVER_PATH = (
    PROJECT_ROOT
    / "data"
    / "silver"
    / "silver_master.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "gold"
    / "exports"
    / "ml"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


TARGET = "default_risk"


# ---------------------------------------------------------------------
# Candidate features
# ---------------------------------------------------------------------

CANDIDATE_FEATURES = [
    # Behaviour
    "num_times_delinquent",
    "num_times_30p_dpd",
    "num_times_60p_dpd",
    "delinquency_score",
    "missed_payment_ratio",

    # Portfolio
    "Total_TL",
    "Tot_Active_TL",
    "active_loan_ratio",
    "loan_type_diversity",
    "credit_history_months",

    # Demographics / income
    "AGE",
    "NETMONTHLYINCOME",

    # Credit seeking behaviour
    "enq_L3m",
    "enq_L6m",
    "enq_L12m",
    "tot_enq",

    # Product mix
    "Gold_TL",
    "Home_TL",
    "PL_TL",
    "CC_TL",
    "Auto_TL",

    # Activity
    "recently_active",

    # Missingness indicators
    "has_cc",
    "has_pl",
]


# ---------------------------------------------------------------------
# Explicit leakage exclusions
# ---------------------------------------------------------------------

LEAKAGE_COLUMNS = {
    # Target
    "default_risk",
    "risk_target",

    # Original label
    "Approved_Flag",

    # Direct target transformations
    "risk_grade",
    "risk_band",

    # CIBIL-derived fields are retained in Silver for analytics, but are
    # deliberately excluded from the production ML feature matrix.
    # Credit_Score is directly related to the target construction, while
    # cibil_band is derived from Credit_Score.
    "Credit_Score",
    "cibil_band",

    # Other obvious post-label / administrative identifiers
    "PROSPECTID",
    "_source",
    "_ingested_at",
}


class CreditRiskModelPipeline:
    """Train, evaluate and explain credit-risk classifiers."""

    # -----------------------------------------------------------------
    # LOAD
    # -----------------------------------------------------------------

    def load_data(self) -> pd.DataFrame:

        print("\n[1/8] Loading Silver dataset...")

        if not SILVER_PATH.exists():
            raise FileNotFoundError(
                f"Silver dataset not found:\n{SILVER_PATH}\n\n"
                "Run the Silver transformation first."
            )

        df = pd.read_parquet(
            SILVER_PATH
        )

        print(
            f"  Shape: {df.shape}"
        )

        if TARGET not in df.columns:
            raise ValueError(
                f"Required target '{TARGET}' "
                "does not exist."
            )

        if "risk_target" in df.columns:
            raise ValueError(
                "Legacy target 'risk_target' exists. "
                "Remove it before ML."
            )

        return df

    # -----------------------------------------------------------------
    # FEATURE AUDIT
    # -----------------------------------------------------------------

    def select_features(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, list[str]]:

        print("\n[2/8] Building leakage-safe feature matrix...")

        available = [
            col
            for col in CANDIDATE_FEATURES
            if col in df.columns
        ]

        missing = [
            col
            for col in CANDIDATE_FEATURES
            if col not in df.columns
        ]

        if missing:
            print(
                f"  Optional features not present: {missing}"
            )

        # Defensive leakage audit.
        leaked = [
            col
            for col in available
            if col in LEAKAGE_COLUMNS
        ]

        if leaked:
            raise ValueError(
                f"Leakage columns entered the feature matrix: {leaked}"
            )

        if not available:
            raise ValueError(
                "No usable ML features were found."
            )

        X = df[available].copy()

        # Remove features with no variation.
        constant_columns = [
            col
            for col in X.columns
            if X[col].nunique(dropna=False) <= 1
        ]

        if constant_columns:
            print(
                f"  Removing constant features: "
                f"{constant_columns}"
            )

            X = X.drop(
                columns=constant_columns
            )

            available = [
                col
                for col in available
                if col not in constant_columns
            ]

        print(
            f"  Final feature count: {len(available)}"
        )

        print(
            f"  Explicitly excluded Credit_Score: "
            f"{'Credit_Score' not in available}"
        )

        print(
            f"  Explicitly excluded cibil_band: "
            f"{'cibil_band' not in available}"
        )

        print(
            f"  Excluded target-derived fields: "
            f"{all(c not in available for c in ['risk_grade', 'risk_band', 'Approved_Flag'])}"
        )

        return X, available

    # -----------------------------------------------------------------
    # SPLIT
    # -----------------------------------------------------------------

    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ):
        print("\n[3/8] Creating stratified train/test split...")

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42,
                stratify=y,
            )
        )

        print(
            f"  Train: {X_train.shape}"
        )

        print(
            f"  Test : {X_test.shape}"
        )

        print(
            f"  Train positive rate: "
            f"{y_train.mean() * 100:.2f}%"
        )

        print(
            f"  Test positive rate: "
            f"{y_test.mean() * 100:.2f}%"
        )

        return (
            X_train,
            X_test,
            y_train,
            y_test,
        )

    # -----------------------------------------------------------------
    # PREPROCESSOR
    # -----------------------------------------------------------------

    def build_preprocessor(
        self,
        X: pd.DataFrame,
    ) -> ColumnTransformer:

        numeric_columns = X.select_dtypes(
            include=[np.number]
        ).columns.tolist()

        categorical_columns = X.select_dtypes(
            include=["object", "string", "category"]
        ).columns.tolist()

        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                        add_indicator=True,
                    ),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent",
                    ),
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )

        transformers = []

        if numeric_columns:
            transformers.append(
                (
                    "numeric",
                    numeric_pipeline,
                    numeric_columns,
                )
            )

        if categorical_columns:
            transformers.append(
                (
                    "categorical",
                    categorical_pipeline,
                    categorical_columns,
                )
            )

        return ColumnTransformer(
            transformers=transformers,
            remainder="drop",
        )

    # -----------------------------------------------------------------
    # METRICS
    # -----------------------------------------------------------------

    @staticmethod
    def calculate_metrics(
        y_true,
        probabilities,
        threshold: float = 0.50,
    ) -> dict:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        ).ravel()

        return {
            "threshold": float(threshold),
            "accuracy": float(
                accuracy_score(
                    y_true,
                    predictions,
                )
            ),
            "precision": float(
                precision_score(
                    y_true,
                    predictions,
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    y_true,
                    predictions,
                    zero_division=0,
                )
            ),
            "f1": float(
                f1_score(
                    y_true,
                    predictions,
                    zero_division=0,
                )
            ),
            "roc_auc": float(
                roc_auc_score(
                    y_true,
                    probabilities,
                )
            ),
            "pr_auc": float(
                average_precision_score(
                    y_true,
                    probabilities,
                )
            ),
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        }

    # -----------------------------------------------------------------
    # CROSS VALIDATION
    # -----------------------------------------------------------------

    def cross_validate_models(
        self,
        X_train,
        y_train,
        preprocessor,
        scale_pos_weight,
    ):

        print("\n[4/8] Comparing baseline and XGBoost...")

        logistic = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        )

        xgb_model = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "classifier",
                    xgb.XGBClassifier(
                        n_estimators=300,
                        max_depth=5,
                        learning_rate=0.05,
                        subsample=0.80,
                        colsample_bytree=0.80,
                        min_child_weight=3,
                        reg_lambda=2.0,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        scale_pos_weight=scale_pos_weight,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42,
        )

        scoring = {
            "roc_auc": "roc_auc",
            "pr_auc": "average_precision",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "accuracy": "accuracy",
        }

        results = {}

        for name, model in [
            ("logistic_regression", logistic),
            ("xgboost", xgb_model),
        ]:

            print(
                f"\n  Cross-validating {name}..."
            )

            cv_results = cross_validate(
                model,
                X_train,
                y_train,
                cv=cv,
                scoring=scoring,
                n_jobs=1,
            )

            summary = {}

            for metric in scoring:
                values = cv_results[
                    f"test_{metric}"
                ]

                summary[metric] = {
                    "mean": float(
                        np.mean(values)
                    ),
                    "std": float(
                        np.std(values)
                    ),
                }

            results[name] = summary

            print(
                f"    ROC-AUC: "
                f"{summary['roc_auc']['mean']:.4f} "
                f"+/- "
                f"{summary['roc_auc']['std']:.4f}"
            )

            print(
                f"    PR-AUC : "
                f"{summary['pr_auc']['mean']:.4f}"
            )

            print(
                f"    Recall : "
                f"{summary['recall']['mean']:.4f}"
            )

            print(
                f"    F1     : "
                f"{summary['f1']['mean']:.4f}"
            )

        return (
            logistic,
            xgb_model,
            results,
        )

    # -----------------------------------------------------------------
    # THRESHOLD ANALYSIS
    # -----------------------------------------------------------------

    def threshold_analysis(
        self,
        y_test,
        probabilities,
    ) -> pd.DataFrame:

        rows = []

        for threshold in np.arange(
            0.20,
            0.81,
            0.05,
        ):

            metrics = self.calculate_metrics(
                y_test,
                probabilities,
                threshold=float(
                    round(threshold, 2)
                ),
            )

            rows.append(metrics)

        return pd.DataFrame(rows)

    # -----------------------------------------------------------------
    # TRAIN FINAL MODEL
    # -----------------------------------------------------------------

    def train_final_model(
        self,
        model,
        X_train,
        y_train,
    ):

        print("\n[5/8] Training final XGBoost model...")

        model.fit(
            X_train,
            y_train,
        )

        return model

    # -----------------------------------------------------------------
    # FEATURE NAMES
    # -----------------------------------------------------------------

    @staticmethod
    def get_feature_names(
        fitted_pipeline,
    ) -> np.ndarray:

        preprocessor = fitted_pipeline.named_steps[
            "preprocessor"
        ]

        try:
            return preprocessor.get_feature_names_out()

        except Exception:
            return np.array(
                [
                    f"feature_{i}"
                    for i in range(
                        fitted_pipeline[
                            "preprocessor"
                        ].transform(
                            pd.DataFrame()
                        ).shape[1]
                    )
                ]
            )

    # -----------------------------------------------------------------
    # SHAP
    # -----------------------------------------------------------------

    def generate_shap(
        self,
        fitted_model,
        X_test,
        max_rows: int = 2000,
    ):

        if not SHAP_AVAILABLE:

            print(
                "\n  ⚠ SHAP not installed. "
                "Skipping SHAP explanation."
            )

            return

        print(
            "\n[7/8] Generating SHAP explanations..."
        )

        preprocessor = fitted_model.named_steps[
            "preprocessor"
        ]

        classifier = fitted_model.named_steps[
            "classifier"
        ]

        X_sample = X_test.head(
            min(max_rows, len(X_test))
        )

        transformed = (
            preprocessor.transform(
                X_sample
            )
        )

        feature_names = (
            preprocessor
            .get_feature_names_out()
        )

        try:

            explainer = shap.TreeExplainer(
                classifier
            )

            shap_values = explainer.shap_values(
                transformed
            )

            # Modern SHAP may return an Explanation
            # or ndarray depending on installed version.
            if hasattr(
                shap_values,
                "values",
            ):
                values = shap_values.values
            else:
                values = shap_values

            if isinstance(
                values,
                list,
            ):
                values = values[-1]

            mean_abs = np.mean(
                np.abs(values),
                axis=0,
            )

            importance = pd.DataFrame(
                {
                    "feature": feature_names,
                    "mean_abs_shap": mean_abs,
                }
            ).sort_values(
                "mean_abs_shap",
                ascending=False,
            )

            importance.to_csv(
                OUTPUT_DIR
                / "shap_feature_importance.csv",
                index=False,
            )

            print(
                "\n  Top SHAP drivers:"
            )

            print(
                importance.head(15)
                .to_string(index=False)
            )

        except Exception as exc:

            print(
                f"  ⚠ SHAP generation failed: "
                f"{exc}"
            )

    # -----------------------------------------------------------------
    # MONITORING BASELINE
    # -----------------------------------------------------------------

    @staticmethod
    def build_monitoring_baseline(
        X_train: pd.DataFrame,
    ) -> dict:

        baseline = {}

        for col in X_train.columns:

            series = X_train[col]

            if pd.api.types.is_numeric_dtype(
                series
            ):

                numeric = pd.to_numeric(
                    series,
                    errors="coerce",
                )

                baseline[col] = {
                    "type": "numeric",
                    "missing_rate": float(
                        numeric.isna().mean()
                    ),
                    "mean": float(
                        numeric.mean()
                    )
                    if numeric.notna().any()
                    else None,
                    "std": float(
                        numeric.std()
                    )
                    if numeric.notna().any()
                    else None,
                    "q01": float(
                        numeric.quantile(0.01)
                    )
                    if numeric.notna().any()
                    else None,
                    "q25": float(
                        numeric.quantile(0.25)
                    )
                    if numeric.notna().any()
                    else None,
                    "median": float(
                        numeric.median()
                    )
                    if numeric.notna().any()
                    else None,
                    "q75": float(
                        numeric.quantile(0.75)
                    )
                    if numeric.notna().any()
                    else None,
                    "q99": float(
                        numeric.quantile(0.99)
                    )
                    if numeric.notna().any()
                    else None,
                }

            else:

                distribution = (
                    series
                    .astype("string")
                    .fillna("__MISSING__")
                    .value_counts(
                        normalize=True
                    )
                    .head(20)
                    .to_dict()
                )

                baseline[col] = {
                    "type": "categorical",
                    "missing_rate": float(
                        series.isna().mean()
                    ),
                    "top_distribution": {
                        str(k): float(v)
                        for k, v in distribution.items()
                    },
                }

        return baseline

    # -----------------------------------------------------------------
    # RUN
    # -----------------------------------------------------------------

    def run(self):

        print("=" * 70)
        print(" INDIA CREDIT RISK — MODELING PIPELINE")
        print("=" * 70)

        df = self.load_data()

        X, features = self.select_features(
            df
        )

        y = df[TARGET].astype(int)

        X_train, X_test, y_train, y_test = (
            self.split_data(
                X,
                y,
            )
        )

        preprocessor = (
            self.build_preprocessor(
                X_train
            )
        )

        # Calculate imbalance ONLY from training data.
        positives = int(
            y_train.sum()
        )

        negatives = int(
            len(y_train) - positives
        )

        scale_pos_weight = (
            negatives / positives
            if positives > 0
            else 1.0
        )

        print(
            f"\n  Training negatives: {negatives:,}"
        )

        print(
            f"  Training positives: {positives:,}"
        )

        print(
            f"  scale_pos_weight: "
            f"{scale_pos_weight:.4f}"
        )

        (
            logistic_model,
            xgb_model,
            cv_results,
        ) = self.cross_validate_models(
            X_train,
            y_train,
            preprocessor,
            scale_pos_weight,
        )

        # Train both models so we can make a measured comparison.
        print(
            "\n  Fitting logistic-regression baseline..."
        )

        logistic_model.fit(
            X_train,
            y_train,
        )

        print(
            "  Fitting XGBoost..."
        )

        xgb_model = self.train_final_model(
            xgb_model,
            X_train,
            y_train,
        )

        # -------------------------------------------------------------
        # Test-set evaluation
        # -------------------------------------------------------------

        print(
            "\n[6/8] Evaluating on untouched test set..."
        )

        logistic_prob = (
            logistic_model
            .predict_proba(X_test)[:, 1]
        )

        xgb_prob = (
            xgb_model
            .predict_proba(X_test)[:, 1]
        )

        logistic_metrics = (
            self.calculate_metrics(
                y_test,
                logistic_prob,
            )
        )

        xgb_metrics = (
            self.calculate_metrics(
                y_test,
                xgb_prob,
            )
        )

        print(
            "\n  Logistic Regression:"
        )

        print(
            json.dumps(
                logistic_metrics,
                indent=2,
            )
        )

        print(
            "\n  XGBoost:"
        )

        print(
            json.dumps(
                xgb_metrics,
                indent=2,
            )
        )

        # -------------------------------------------------------------
        # Threshold analysis
        # -------------------------------------------------------------

        threshold_df = (
            self.threshold_analysis(
                y_test,
                xgb_prob,
            )
        )

        threshold_df.to_csv(
            OUTPUT_DIR
            / "threshold_analysis.csv",
            index=False,
        )

        # -------------------------------------------------------------
        # Classification report
        # -------------------------------------------------------------

        xgb_predictions = (
            xgb_prob >= 0.50
        ).astype(int)

        report = classification_report(
            y_test,
            xgb_predictions,
            output_dict=True,
            zero_division=0,
        )

        with open(
            OUTPUT_DIR
            / "classification_report.json",
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                report,
                f,
                indent=2,
            )

        # -------------------------------------------------------------
        # Confusion matrix
        # -------------------------------------------------------------

        cm = confusion_matrix(
            y_test,
            xgb_predictions,
            labels=[0, 1],
        )

        np.savetxt(
            OUTPUT_DIR
            / "confusion_matrix.csv",
            cm,
            delimiter=",",
            fmt="%d",
        )

        # -------------------------------------------------------------
        # SHAP
        # -------------------------------------------------------------

        self.generate_shap(
            xgb_model,
            X_test,
        )

        # -------------------------------------------------------------
        # Monitoring baseline
        # -------------------------------------------------------------

        monitoring = (
            self.build_monitoring_baseline(
                X_train
            )
        )

        with open(
            OUTPUT_DIR
            / "monitoring_baseline.json",
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                monitoring,
                f,
                indent=2,
            )

        # -------------------------------------------------------------
        # Model bundle
        # -------------------------------------------------------------

        model_bundle = {
            "model": xgb_model,
            "target": TARGET,
            "features": features,
            "excluded_leakage_columns": sorted(
                LEAKAGE_COLUMNS
            ),
            "decision_threshold": 0.50,
            "target_definition": {
                "0": "P1/P2 low-risk classification proxy",
                "1": "P3/P4 high-risk classification proxy",
            },
            "credit_score_excluded": True,
            "cibil_band_excluded": True,
            "retained_but_excluded_analytical_fields": [
                "Credit_Score",
                "cibil_band",
            ],
            "trained_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        model_path = (
            PROCESSED_DIR
            / "credit_risk_xgboost.pkl"
        )

        with open(
            model_path,
            "wb",
        ) as f:

            pickle.dump(
                model_bundle,
                f,
            )

        # -------------------------------------------------------------
        # Save model metadata
        # -------------------------------------------------------------

        metadata = {
            "target": TARGET,
            "target_type": (
                "binary high-risk classification proxy"
            ),
            "target_mapping": {
                "P1": 0,
                "P2": 0,
                "P3": 1,
                "P4": 1,
            },
            "features": features,
            "excluded_leakage_columns": sorted(
                LEAKAGE_COLUMNS
            ),
            "credit_score_excluded": True,
            "cibil_band_excluded": True,
            "retained_but_excluded_analytical_fields": [
                "Credit_Score",
                "cibil_band",
            ],
            "train_rows": int(
                len(X_train)
            ),
            "test_rows": int(
                len(X_test)
            ),
            "train_positive_rate": float(
                y_train.mean()
            ),
            "test_positive_rate": float(
                y_test.mean()
            ),
            "scale_pos_weight": float(
                scale_pos_weight
            ),
            "cross_validation": cv_results,
            "test_metrics": {
                "logistic_regression": logistic_metrics,
                "xgboost": xgb_metrics,
            },
            "interpretation": {
                "accuracy": (
                    "Overall classification correctness; "
                    "not sufficient as the sole credit-risk metric."
                ),
                "precision": (
                    "Among borrowers classified high-risk, "
                    "the fraction actually labelled high-risk."
                ),
                "recall": (
                    "Among labelled high-risk borrowers, "
                    "the fraction identified by the model."
                ),
                "f1": (
                    "Harmonic mean of precision and recall."
                ),
                "roc_auc": (
                    "Ranking/discrimination across probability thresholds."
                ),
                "pr_auc": (
                    "Precision-recall performance; particularly useful "
                    "when the positive class is less frequent."
                ),
            },
        }

        with open(
            OUTPUT_DIR
            / "model_metadata.json",
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                metadata,
                f,
                indent=2,
            )

        print(
            "\n[8/8] Saving artifacts..."
        )

        print(
            f"  ✓ Model: {model_path}"
        )

        print(
            f"  ✓ Metadata: "
            f"{OUTPUT_DIR / 'model_metadata.json'}"
        )

        print(
            f"  ✓ Threshold analysis: "
            f"{OUTPUT_DIR / 'threshold_analysis.csv'}"
        )

        print(
            f"  ✓ Monitoring baseline: "
            f"{OUTPUT_DIR / 'monitoring_baseline.json'}"
        )

        print(
            "\n" + "=" * 70
        )

        print(
            " MODELING PIPELINE COMPLETE"
        )

        print(
            "=" * 70
        )

        return {
            "model": xgb_model,
            "metrics": xgb_metrics,
            "cv_results": cv_results,
            "features": features,
        }