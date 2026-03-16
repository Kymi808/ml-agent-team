"""Feature Engineering Agent — imputation, encoding, scaling, feature selection, train/test split."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ..core.base_agent import BaseAgent
from ..core.messages import AgentMessage
from ..core.types import PipelineStage, ProblemType


class FeatureEngineeringAgent(BaseAgent):
    """Handles data preprocessing: imputation, encoding, scaling, and train/test splitting."""

    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.FEATURE_ENGINEERING

    @property
    def description(self) -> str:
        return (
            "Preprocesses data with imputation, encoding, scaling, and feature selection, "
            "then splits into train/test sets"
        )

    @property
    def dependencies(self) -> list[PipelineStage]:
        return [PipelineStage.EDA]

    async def execute(self) -> AgentMessage:
        df = self.state.raw_data.copy()
        profile = self.state.data_profile
        target = self.state.problem.target_column

        self.logger.info("starting_feature_engineering", columns=len(df.columns))

        # 1. Handle missing values
        df = self._impute_missing(df, profile)

        # 2. Encode categorical variables
        df, encoding_maps = self._encode_categoricals(df, profile, target)
        self.state.encoding_maps = encoding_maps

        # 3. Scale numeric features
        df, feature_names = self._scale_features(df, profile, target)

        # 4. Store processed data
        self.state.processed_data = df
        self.state.feature_names = feature_names

        # 5. Train/test split
        test_size = self.config.params.get("test_size", 0.2)
        random_state = self.config.params.get("random_state", 42)
        self._split_data(df, target, test_size, random_state)

        self.logger.info(
            "feature_engineering_complete",
            features=len(feature_names),
            train_size=len(self.state.X_train),
            test_size=len(self.state.X_test),
        )

        return self._result_message({
            "n_features": len(feature_names),
            "train_size": len(self.state.X_train),
            "test_size": len(self.state.X_test),
            "imputed_columns": sum(
                1 for v in profile.missing_counts.values() if v > 0
            ),
            "encoded_columns": len(encoding_maps),
        })

    def _impute_missing(self, df: pd.DataFrame, profile: Any) -> pd.DataFrame:
        """Impute missing values based on column type."""
        for col in profile.numeric_columns:
            if df[col].isnull().any():
                # Use median for skewed distributions, mean otherwise
                skew = abs(df[col].skew())
                if skew > 1:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mean())

        for col in profile.categorical_columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "unknown")

        return df

    def _encode_categoricals(
        self, df: pd.DataFrame, profile: Any, target: str | None
    ) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
        """Encode categorical variables."""
        encoding_maps: dict[str, dict[str, int]] = {}
        cols_to_encode = [
            c for c in profile.categorical_columns if c != target
        ]

        for col in cols_to_encode:
            n_unique = df[col].nunique()

            if n_unique <= 2:
                # Binary: label encode
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoding_maps[col] = {
                    str(cls): int(i) for i, cls in enumerate(le.classes_)
                }
            elif n_unique <= 10:
                # Low cardinality: one-hot encode
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            else:
                # High cardinality: label encode (could use target encoding in production)
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoding_maps[col] = {
                    str(cls): int(i) for i, cls in enumerate(le.classes_)
                }

        # Encode target if categorical
        if target and target in df.columns and df[target].dtype == "object":
            le = LabelEncoder()
            df[target] = le.fit_transform(df[target].astype(str))
            encoding_maps[target] = {
                str(cls): int(i) for i, cls in enumerate(le.classes_)
            }

        return df, encoding_maps

    def _scale_features(
        self, df: pd.DataFrame, profile: Any, target: str | None
    ) -> tuple[pd.DataFrame, list[str]]:
        """Scale numeric features using StandardScaler."""
        feature_cols = [c for c in df.columns if c != target]
        numeric_features = [c for c in feature_cols if c in profile.numeric_columns]

        if numeric_features:
            scaler = StandardScaler()
            df[numeric_features] = scaler.fit_transform(df[numeric_features])

        return df, feature_cols

    def _split_data(
        self,
        df: pd.DataFrame,
        target: str | None,
        test_size: float,
        random_state: int,
    ) -> None:
        """Split data into train and test sets."""
        if target is None or target not in df.columns:
            self.logger.warning("no_target_column", target=target)
            return

        feature_cols = [c for c in df.columns if c != target]
        X = df[feature_cols]
        y = df[target]

        # Use stratification for classification problems
        stratify = None
        problem_type = self.state.problem.problem_type
        if problem_type in (
            ProblemType.BINARY_CLASSIFICATION,
            ProblemType.MULTICLASS_CLASSIFICATION,
        ):
            if y.nunique() <= 50:  # Only stratify if reasonable number of classes
                stratify = y

        self.state.X_train, self.state.X_test, self.state.y_train, self.state.y_test = (
            train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=stratify
            )
        )
