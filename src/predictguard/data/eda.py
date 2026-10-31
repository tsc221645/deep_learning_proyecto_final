"""Exploratory analysis for NASA C-MAPSS FD001."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from predictguard.data.validate import (
    COLUMN_NAMES,
    RUL_FILE,
    TEST_FILE,
    TRAIN_FILE,
    resolve_data_dir,
    validate_fd001,
)

ID_COLUMNS = ["unit_id", "cycle"]
SETTING_COLUMNS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLUMNS = [f"sensor_{index}" for index in range(1, 22)]
FEATURE_COLUMNS = SETTING_COLUMNS + SENSOR_COLUMNS
NEAR_CONSTANT_MODAL_FRACTION = 0.98
STRONG_CORRELATION_THRESHOLD = 0.95
OUTLIER_IQR_MULTIPLIER = 1.5


def _read_frames(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    train = pd.read_csv(
        data_dir / TRAIN_FILE, sep=r"\s+", header=None, names=COLUMN_NAMES
    )
    test = pd.read_csv(
        data_dir / TEST_FILE, sep=r"\s+", header=None, names=COLUMN_NAMES
    )
    rul = pd.read_csv(data_dir / RUL_FILE, sep=r"\s+", header=None).iloc[:, 0]
    return train, test, rul


def _column_summary(
    frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for dataset_name, frame in frames.items():
        for column in frame.columns:
            values = frame[column]
            numeric_values = pd.to_numeric(values, errors="coerce")
            rows.append(
                {
                    "dataset": dataset_name,
                    "column": column,
                    "dtype": str(values.dtype),
                    "missing": int(values.isna().sum()),
                    "infinite": (
                        int(np.isinf(numeric_values.to_numpy(dtype=float)).sum())
                        if pd.api.types.is_numeric_dtype(values)
                        else 0
                    ),
                    "median": values.median(),
                    "std": values.std(),
                    "min": values.min(),
                    "max": values.max(),
                    "unique": int(values.nunique(dropna=False)),
                }
            )
    return pd.DataFrame(rows)


def _duration_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    duration_frames = []
    for dataset_name, frame in frames.items():
        durations = frame.groupby("unit_id", sort=True)["cycle"].agg(
            cycles="size", max_cycle="max"
        )
        durations["dataset"] = dataset_name
        durations["unit_id"] = durations.index.astype(int)
        duration_frames.append(durations.reset_index(drop=True))
    return pd.concat(duration_frames, ignore_index=True)[
        ["dataset", "unit_id", "cycles", "max_cycle"]
    ]


def _sensor_summary(train: pd.DataFrame) -> pd.DataFrame:
    sensor_frame = train[SENSOR_COLUMNS]
    summary = pd.DataFrame(
        {
            "sensor": SENSOR_COLUMNS,
            "variance": sensor_frame.var().to_numpy(),
            "std": sensor_frame.std().to_numpy(),
            "unique": sensor_frame.nunique().to_numpy(),
            "modal_fraction": [
                sensor_frame[sensor].value_counts(normalize=True).iloc[0]
                for sensor in SENSOR_COLUMNS
            ],
        }
    )

    correlations_by_engine = []
    for _, group in train.groupby("unit_id", sort=True):
        correlations_by_engine.append(
            {
                sensor: (
                    group[sensor].corr(group["cycle"], method="spearman")
                    if group[sensor].nunique() > 1
                    else np.nan
                )
                for sensor in SENSOR_COLUMNS
            }
        )
    per_unit_correlations = pd.DataFrame(correlations_by_engine)
    summary["median_abs_unit_spearman_cycle"] = (
        per_unit_correlations.abs().median().reindex(SENSOR_COLUMNS).to_numpy()
    )
    summary["constant"] = summary["unique"].eq(1)
    summary["near_constant_candidate"] = (
        ~summary["constant"]
        & summary["modal_fraction"].ge(NEAR_CONSTANT_MODAL_FRACTION)
    )
    return summary


def _correlation_table(train: pd.DataFrame) -> pd.DataFrame:
    correlation = train[SENSOR_COLUMNS].corr()
    rows = [
        {"sensor_a": sensor_a, "sensor_b": sensor_b, "correlation": correlation.loc[sensor_a, sensor_b]}
        for index, sensor_a in enumerate(SENSOR_COLUMNS)
        for sensor_b in SENSOR_COLUMNS[index + 1 :]
        if abs(correlation.loc[sensor_a, sensor_b]) >= STRONG_CORRELATION_THRESHOLD
    ]
    return pd.DataFrame(rows, columns=["sensor_a", "sensor_b", "correlation"])


def _outlier_table(train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sensor in SENSOR_COLUMNS:
        values = train[sensor]
        lower_quartile, upper_quartile = values.quantile([0.25, 0.75])
        spread = upper_quartile - lower_quartile
        lower_bound = lower_quartile - OUTLIER_IQR_MULTIPLIER * spread
        upper_bound = upper_quartile + OUTLIER_IQR_MULTIPLIER * spread
        outliers = (values < lower_bound) | (values > upper_bound)
        rows.append(
            {
                "sensor": sensor,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "count": int(outliers.sum()),
                "fraction": float(outliers.mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("fraction", ascending=False)


def _train_test_comparison(
    train: pd.DataFrame, test: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for sensor in SENSOR_COLUMNS:
        train_mean = float(train[sensor].mean())
        train_std = float(train[sensor].std())
        for dataset_name, frame in (("train", train), ("test", test)):
            rows.append(
                {
                    "dataset": dataset_name,
                    "sensor": sensor,
                    "mean": float(frame[sensor].mean()),
                    "median": float(frame[sensor].median()),
                    "std": float(frame[sensor].std()),
                    "mean_in_train_std_units": (
                        float((frame[sensor].mean() - train_mean) / train_std)
                        if train_std > 0
                        else 0.0
                    ),
                }
            )
    return pd.DataFrame(rows)


def _rul_balance(
    train: pd.DataFrame, test_rul: pd.Series, window_size: int
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    last_cycle = train.groupby("unit_id")["cycle"].transform("max")
    training_rul = (last_cycle - train["cycle"]).astype(int)
    max_window_endpoints = (
        train.groupby("unit_id")["cycle"].transform("size") - window_size + 1
    ).clip(lower=0)
    has_window_endpoint = (
        train.groupby("unit_id").cumcount() >= (window_size - 1)
    )
    window_rul = training_rul.loc[has_window_endpoint]

    bin_edges = [-1, 30, 60, 90, 120, 150, np.inf]
    bin_labels = ["0-30", "31-60", "61-90", "91-120", "121-150", "151+"]
    train_bins = pd.cut(training_rul, bins=bin_edges, labels=bin_labels)
    window_bins = pd.cut(window_rul, bins=bin_edges, labels=bin_labels)
    test_bins = pd.cut(test_rul, bins=bin_edges, labels=bin_labels)
    rows = []
    for source, bins in (
        ("train_all_cycles", train_bins),
        ("train_window_endpoints", window_bins),
        ("test_official_last_cycle_only", test_bins),
    ):
        counts = bins.value_counts(sort=False)
        total = int(counts.sum())
        for label in bin_labels:
            count = int(counts.get(label, 0))
            rows.append(
                {
                    "source": source,
                    "rul_bin_cycles": label,
                    "count": count,
                    "fraction": count / total if total else 0.0,
                    "total": total,
                }
            )
    window_info = pd.Series(
        {
            "window_size": window_size,
            "train_windows": int(max_window_endpoints.groupby(train["unit_id"]).first().sum()),
            "train_window_endpoints_with_rul_le_30": int((window_rul <= 30).sum()),
            "train_windows_per_engine_min": int(
                max_window_endpoints.groupby(train["unit_id"]).first().min()
            ),
            "train_windows_per_engine_max": int(
                max_window_endpoints.groupby(train["unit_id"]).first().max()
            ),
        }
    )
    return pd.DataFrame(rows), training_rul, window_info


def _sequence_diagnostics(frames: dict[str, pd.DataFrame]) -> dict[str, dict[str, int | bool]]:
    diagnostics = {}
    for dataset_name, frame in frames.items():
        cycle_differences = frame.groupby("unit_id", sort=False)["cycle"].diff()
        diagnostics[dataset_name] = {
            "duplicate_unit_cycle_pairs": int(
                frame.duplicated(["unit_id", "cycle"]).sum()
            ),
            "non_increasing_cycle_steps": int(
                (cycle_differences.dropna() <= 0).sum()
            ),
            "non_consecutive_cycle_steps": int(
                (cycle_differences.dropna() != 1).sum()
            ),
            "all_cycles_strictly_increasing_in_file_order": bool(
                (cycle_differences.dropna() > 0).all()
            ),
        }
    return diagnostics


def _save_duration_plot(durations: pd.DataFrame, output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(9, 5))
    for dataset_name, color in (("train", "#2563eb"), ("test", "#f97316")):
        values = durations.loc[durations["dataset"] == dataset_name, "cycles"]
        axis.hist(values, bins=15, alpha=0.6, label=dataset_name, color=color)
    axis.set(title="Ciclos observados por motor", xlabel="Ciclos", ylabel="Motores")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_trajectory_plot(
    train: pd.DataFrame, durations: pd.DataFrame, sensor_summary: pd.DataFrame, output_path: Path
) -> list[int]:
    train_durations = durations.loc[durations["dataset"] == "train"].sort_values(
        "cycles"
    )
    quantile_positions = [0.1, 0.3, 0.5, 0.7, 0.9]
    positions = [
        min(int(round(quantile * (len(train_durations) - 1))), len(train_durations) - 1)
        for quantile in quantile_positions
    ]
    engine_ids = (
        train_durations.iloc[positions]["unit_id"].drop_duplicates().astype(int).tolist()
    )
    selected_sensors = (
        sensor_summary.loc[~sensor_summary["constant"]]
        .sort_values("median_abs_unit_spearman_cycle", ascending=False)
        .head(4)["sensor"]
        .tolist()
    )
    if not selected_sensors:
        selected_sensors = SENSOR_COLUMNS[:4]

    fig, axes = plt.subplots(
        len(selected_sensors), 1, figsize=(11, 3 * len(selected_sensors)), sharex=True
    )
    axes = np.atleast_1d(axes)
    for axis, sensor in zip(axes, selected_sensors):
        for engine_id in engine_ids:
            trajectory = train.loc[train["unit_id"] == engine_id]
            axis.plot(
                trajectory["cycle"],
                trajectory[sensor],
                linewidth=1,
                alpha=0.8,
                label=f"Motor {engine_id}",
            )
        axis.set_ylabel(sensor)
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Ciclo")
    axes[0].set_title("Sensores con tendencia temporal en motores representativos")
    axes[0].legend(ncol=5, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return engine_ids


def _save_sensor_diagnostics(
    sensor_summary: pd.DataFrame,
    correlation: pd.DataFrame,
    output_dir: Path,
) -> None:
    ordered = sensor_summary.sort_values("variance", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    axes[0].barh(ordered["sensor"], ordered["variance"], color="#2563eb")
    axes[0].set_xscale("symlog", linthresh=1e-12)
    axes[0].set(title="Varianza por sensor (train)", xlabel="Varianza, escala symlog")
    axes[1].barh(
        sensor_summary["sensor"],
        sensor_summary["unique"],
        color=[
            "#dc2626" if is_constant else "#f97316" if near_constant else "#16a34a"
            for is_constant, near_constant in zip(
                sensor_summary["constant"], sensor_summary["near_constant_candidate"]
            )
        ],
    )
    axes[1].set(title="Valores únicos por sensor (train)", xlabel="Valores únicos")
    fig.tight_layout()
    fig.savefig(output_dir / "eda_sensor_variance.png", dpi=160)
    plt.close(fig)

    train_corr = correlation.attrs.get("correlation_matrix")
    if train_corr is not None:
        fig, axis = plt.subplots(figsize=(10, 8))
        image = axis.imshow(train_corr, vmin=-1, vmax=1, cmap="coolwarm")
        axis.set_xticks(range(len(SENSOR_COLUMNS)), SENSOR_COLUMNS, rotation=90)
        axis.set_yticks(range(len(SENSOR_COLUMNS)), SENSOR_COLUMNS)
        axis.set_title("Correlación de Pearson entre sensores (train)")
        fig.colorbar(image, ax=axis, label="Correlación")
        fig.tight_layout()
        fig.savefig(output_dir / "eda_sensor_correlations.png", dpi=160)
        plt.close(fig)


def _save_rul_plot(
    training_rul: pd.Series,
    test_rul: pd.Series,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(training_rul, bins=30, color="#2563eb", edgecolor="white")
    axes[0].set(
        title="RUL en ciclos de entrenamiento",
        xlabel="RUL = último ciclo observado del motor - ciclo",
        ylabel="Filas de entrenamiento",
    )
    axes[1].hist(test_rul, bins=20, color="#f97316", edgecolor="white")
    axes[1].set(
        title="RUL oficial de test",
        xlabel="RUL en el último ciclo observado",
        ylabel="Motores de test",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_train_test_plot(
    comparison: pd.DataFrame, output_path: Path
) -> None:
    sensors_per_row = 11
    fig, axes = plt.subplots(2, 1, figsize=(16, 11))
    for row_index, sensor_batch in enumerate(
        (
            SENSOR_COLUMNS[:sensors_per_row],
            SENSOR_COLUMNS[sensors_per_row:],
        )
    ):
        axis = axes[row_index]
        train_stats = (
            comparison.loc[comparison["dataset"] == "train"]
            .set_index("sensor")
            .loc[sensor_batch]
        )
        test_stats = (
            comparison.loc[comparison["dataset"] == "test"]
            .set_index("sensor")
            .loc[sensor_batch]
        )
        positions = np.arange(len(sensor_batch))
        axis.bar(
            positions - 0.18,
            train_stats["mean_in_train_std_units"],
            width=0.36,
            label="train",
            color="#2563eb",
        )
        axis.bar(
            positions + 0.18,
            test_stats["mean_in_train_std_units"],
            width=0.36,
            label="test",
            color="#f97316",
        )
        axis.set_xticks(positions, sensor_batch, rotation=45, ha="right")
        axis.set_ylabel("Media en desviaciones estándar de train")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_title("Comparación descriptiva de sensores train/test")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_segment_plot(train: pd.DataFrame, output_path: Path) -> list[str]:
    frame = train.copy()
    maximum_cycle = frame.groupby("unit_id")["cycle"].transform("max")
    age_fraction = frame["cycle"] / maximum_cycle
    frame["lifecycle_segment"] = pd.cut(
        age_fraction,
        bins=[0, 1 / 3, 2 / 3, 1],
        labels=["inicio", "medio", "final"],
        include_lowest=True,
    )
    segment_medians = frame.groupby(
        "lifecycle_segment", observed=False
    )[SENSOR_COLUMNS].median()
    full_train_std = train[SENSOR_COLUMNS].std().replace(0, np.nan)
    standardized_shift = (
        segment_medians.loc["final"] - segment_medians.loc["inicio"]
    ) / full_train_std
    selected_sensors = (
        standardized_shift.abs().dropna().sort_values(ascending=False).head(6).index.tolist()
    )
    if selected_sensors:
        standardized_segments = (
            segment_medians[selected_sensors] - train[selected_sensors].mean()
        ) / full_train_std[selected_sensors]
        fig, axis = plt.subplots(figsize=(10, 5))
        for sensor in selected_sensors:
            axis.plot(
                ["inicio", "medio", "final"],
                standardized_segments[sensor],
                marker="o",
                label=sensor,
            )
        axis.set(
            title="Cambio de mediana por tercio del ciclo de vida (train)",
            ylabel="Mediana estandarizada con media/desviación de train",
        )
        axis.grid(alpha=0.2)
        axis.legend(ncol=3)
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
    return selected_sensors


def _write_report(
    report_path: Path,
    *,
    durations: pd.DataFrame,
    diagnostics: dict[str, dict[str, int | bool]],
    sensor_summary: pd.DataFrame,
    correlations: pd.DataFrame,
    outliers: pd.DataFrame,
    rul_balance: pd.DataFrame,
    window_info: pd.Series,
    train_test_comparison: pd.DataFrame,
    trajectory_engines: list[int],
    segment_sensors: list[str],
    summary: pd.DataFrame,
) -> None:
    train_durations = durations.loc[durations["dataset"] == "train", "cycles"]
    test_durations = durations.loc[durations["dataset"] == "test", "cycles"]
    constant_sensors = sensor_summary.loc[
        sensor_summary["constant"], "sensor"
    ].tolist()
    near_constant_sensors = sensor_summary.loc[
        sensor_summary["near_constant_candidate"], "sensor"
    ].tolist()
    high_correlation_pairs = [
        f"{row.sensor_a} y {row.sensor_b} ({row.correlation:.3f})"
        for row in correlations.itertuples(index=False)
    ]
    top_outliers = outliers.head(5)
    train_low_rul = rul_balance.loc[
        (rul_balance["source"] == "train_window_endpoints")
        & (rul_balance["rul_bin_cycles"] == "0-30")
    ].iloc[0]
    max_shift = train_test_comparison.assign(
        abs_shift=lambda data: data["mean_in_train_std_units"].abs()
    ).sort_values("abs_shift", ascending=False)
    largest_shifts = [
        f"{row.sensor}: {row.mean_in_train_std_units:.2f}"
        for row in max_shift.loc[max_shift["dataset"] == "test"]
        .head(5)
        .itertuples(index=False)
    ]
    issue_rows = summary.loc[summary["missing"].gt(0) | summary["infinite"].gt(0)]
    issue_text = (
        f"{len(issue_rows)} columnas con faltantes o infinitos."
        if not issue_rows.empty
        else "No se detectaron faltantes ni infinitos en train/test."
    )
    sequence_text = "\n".join(
        f"- **{dataset}**: {facts['duplicate_unit_cycle_pairs']} pares "
        f"(motor, ciclo) duplicados; {facts['non_increasing_cycle_steps']} "
        f"saltos no crecientes; {facts['non_consecutive_cycle_steps']} saltos "
        f"no consecutivos; ciclos estrictamente crecientes en orden de "
        f"archivo: {facts['all_cycles_strictly_increasing_in_file_order']}."
        for dataset, facts in diagnostics.items()
    )
    outlier_text = "\n".join(
        f"- `{row.sensor}`: {row.count} registros "
        f"({row.fraction:.2%}) fuera de 1.5 IQR."
        for row in top_outliers.itertuples(index=False)
    )
    shift_text = (
        ", ".join(f"`{sensor}`" for sensor in largest_shifts)
        if largest_shifts
        else "ninguna"
    )
    correlation_text = (
        "; ".join(high_correlation_pairs)
        if high_correlation_pairs
        else "ningún par superó el umbral"
    )
    report_path.write_text(
        f"""# Hallazgos EDA: NASA C-MAPSS FD001

## Alcance

Se analizaron los archivos originales en train y test. No se modificaron ni
eliminaron registros. Los datos son simulados; cada fila representa un ciclo,
no una fecha u hora real. FD001 contiene una condición operativa y un modo de
degradación (HPC), según el `readme.txt` y el catálogo del conjunto.

## Integridad y duración de trayectorias

- Train: {len(train_durations):,} motores, {int(train_durations.sum()):,} filas.
  Duración mediana {train_durations.median():.0f} ciclos, rango
  {train_durations.min():.0f}-{train_durations.max():.0f}.
- Test: {len(test_durations):,} motores, {int(test_durations.sum()):,} filas.
  Duración mediana {test_durations.median():.1f} ciclos, rango
  {test_durations.min():.0f}-{test_durations.max():.0f}.
- {issue_text}
{sequence_text}

## Sensores

- Constantes en train: {", ".join(f"`{sensor}`" for sensor in constant_sensors) or "ninguno"}.
- Candidatos casi constantes: {", ".join(f"`{sensor}`" for sensor in near_constant_sensors) or "ninguno"}.
  El criterio exploratorio es que un único valor represente al menos
  {NEAR_CONSTANT_MODAL_FRACTION:.0%} de las observaciones; no es una decisión
  automática para excluir variables.
- Pares con correlación de Pearson absoluta >= {STRONG_CORRELATION_THRESHOLD:.2f}:
  {correlation_text}.
- Sensores que se muestran en trayectorias representativas:
  {", ".join(f"`{sensor}`" for sensor in sensor_summary.loc[~sensor_summary["constant"]].sort_values("median_abs_unit_spearman_cycle", ascending=False).head(4)["sensor"])}.
  La selección es visual y no constituye una selección final de features.
- Sensores con mayor cambio entre los tercios inicial y final de la vida:
  {", ".join(f"`{sensor}`" for sensor in segment_sensors) or "sin cambio calculable"}.
- Motores representativos seleccionados por cuantiles de duración:
  {", ".join(map(str, trajectory_engines))}.

## RUL y ventanas

- Se calculó RUL de entrenamiento por fila como
  `máximo ciclo observado del motor - ciclo`.
- Test se describe exclusivamente con los 100 RUL oficiales del último ciclo
  observado por motor. No se asignaron etiquetas a los ciclos parciales de test.
- Tamaño de ventana usado únicamente para el conteo diagnóstico:
  {window_info["window_size"]} ciclos. Train produce
  {window_info["train_windows"]:,} ventanas completas; cada motor produce
  entre {window_info["train_windows_per_engine_min"]:,} y
  {window_info["train_windows_per_engine_max"]:,}. De los extremos de ventana,
  {int(train_low_rul["count"]):,} ({train_low_rul["fraction"]:.1%}) tienen RUL
  <= 30 ciclos. Esto evidencia desbalance de etiquetas según su cercanía al fallo.
- Las etiquetas de test son más escasas que las de train y resumen un único
  ciclo terminal por motor. Ambas distribuciones se grafican por separado y
  no se deben comparar como si fueran muestras equivalentes.

## Comparación descriptiva train/test

Se compararon sensores sin usar el RUL de test. Las medias de test con mayor
desplazamiento, expresadas en desviaciones estándar estimadas solo en train,
son: {shift_text}. Esta visualización es descriptiva y no decide features,
normalización ni ajustes del modelo.

## Valores atípicos

El criterio de cribado fue el rango 1.5 IQR aplicado a cada sensor de train.
Los sensores con mayor proporción de observaciones marcadas fueron:
{outlier_text}

Estos valores se conservan. Antes de filtrarlos se deben inspeccionar sus
trayectorias: pueden ser señales válidas de degradación y su eliminación
podría quitar información predictiva.

## Decisiones para la siguiente fase

1. Conservar los archivos crudos intactos y no corregir valores sin una anomalía
   verificable.
2. Evaluar los sensores constantes y casi constantes usando **solo el
   subconjunto de ajuste** en la fase de preprocesamiento. La lista anterior es
   exploratoria, no una decisión entrenada con validación/test.
3. Ajustar cualquier escalador solo con motores de ajuste.
4. Mantener el test oficial reservado. Sus etiquetas corresponden únicamente al
   último ciclo observado de cada motor.
5. Tratar las diferencias entre motores como trayectorias separadas y nunca
   presentar ciclos como tiempo calendario.

## Archivos generados

- `artifacts/figures/eda_engine_durations.png`
- `artifacts/figures/eda_sensor_trajectories.png`
- `artifacts/figures/eda_sensor_variance.png`
- `artifacts/figures/eda_sensor_correlations.png`
- `artifacts/figures/eda_train_rul_distribution.png`
- `artifacts/figures/eda_train_test_sensor_distributions.png`
- `artifacts/figures/eda_lifecycle_segments.png`
- Tablas CSV en `artifacts/metrics/`.
""",
        encoding="utf-8",
    )


def run_eda(
    data_dir: Path,
    figure_dir: Path,
    metrics_dir: Path,
    report_path: Path,
    *,
    window_size: int = 30,
) -> dict[str, Any]:
    """Analyze FD001 and write plots, diagnostic tables, and findings."""
    if window_size < 1:
        raise ValueError("window_size debe ser mayor que cero.")
    manifest = validate_fd001(data_dir)
    train, test, test_rul = _read_frames(Path(data_dir))
    if manifest["files"][TRAIN_FILE]["rows"] != len(train):
        raise ValueError("Los datos cambiaron después de su validación.")

    figure_dir = Path(figure_dir)
    metrics_dir = Path(metrics_dir)
    report_path = Path(report_path)
    figure_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    frames = {"train": train, "test": test}
    summary = _column_summary(frames)
    durations = _duration_table(frames)
    sensor_summary = _sensor_summary(train)
    correlation_matrix = train[SENSOR_COLUMNS].corr()
    strong_correlations = _correlation_table(train)
    strong_correlations.attrs["correlation_matrix"] = correlation_matrix
    outliers = _outlier_table(train)
    train_test_comparison = _train_test_comparison(train, test)
    rul_balance, training_rul, window_info = _rul_balance(
        train, test_rul, window_size
    )
    diagnostics = _sequence_diagnostics(frames)

    summary.to_csv(metrics_dir / "eda_column_summary.csv", index=False)
    durations.to_csv(metrics_dir / "eda_engine_durations.csv", index=False)
    sensor_summary.to_csv(metrics_dir / "eda_sensor_summary.csv", index=False)
    strong_correlations.to_csv(
        metrics_dir / "eda_strong_correlations.csv", index=False
    )
    outliers.to_csv(metrics_dir / "eda_outliers.csv", index=False)
    train_test_comparison.to_csv(
        metrics_dir / "eda_train_test_comparison.csv", index=False
    )
    rul_balance.to_csv(metrics_dir / "eda_rul_balance.csv", index=False)

    _save_duration_plot(durations, figure_dir / "eda_engine_durations.png")
    trajectory_engines = _save_trajectory_plot(
        train,
        durations,
        sensor_summary,
        figure_dir / "eda_sensor_trajectories.png",
    )
    _save_sensor_diagnostics(sensor_summary, strong_correlations, figure_dir)
    _save_rul_plot(
        training_rul,
        test_rul,
        figure_dir / "eda_train_rul_distribution.png",
    )
    _save_train_test_plot(
        train_test_comparison,
        figure_dir / "eda_train_test_sensor_distributions.png",
    )
    segment_sensors = _save_segment_plot(
        train, figure_dir / "eda_lifecycle_segments.png"
    )
    _write_report(
        report_path,
        durations=durations,
        diagnostics=diagnostics,
        sensor_summary=sensor_summary,
        correlations=strong_correlations,
        outliers=outliers,
        rul_balance=rul_balance,
        window_info=window_info,
        train_test_comparison=train_test_comparison,
        trajectory_engines=trajectory_engines,
        segment_sensors=segment_sensors,
        summary=summary,
    )
    return {
        "column_summary": summary,
        "durations": durations,
        "sensor_summary": sensor_summary,
        "strong_correlations": strong_correlations,
        "outliers": outliers,
        "train_test_comparison": train_test_comparison,
        "rul_balance": rul_balance,
        "window_info": window_info,
        "sequence_diagnostics": diagnostics,
        "trajectory_engines": trajectory_engines,
        "segment_sensors": segment_sensors,
        "report_path": report_path,
        "figure_dir": figure_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ejecutar el EDA de FD001.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config.yaml"),
        help="Ruta a la configuración del proyecto.",
    )
    args = parser.parse_args()
    data_dir = resolve_data_dir(args.config)
    repo_root = Path(args.config).resolve().parent.parent
    with Path(args.config).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    window_size = 30
    if isinstance(config, dict):
        window_size = int(config.get("window", {}).get("size", window_size))
    results = run_eda(
        data_dir,
        repo_root / "artifacts" / "figures",
        repo_root / "artifacts" / "metrics",
        repo_root / "docs" / "eda_findings.md",
        window_size=window_size,
    )
    print(f"Reporte: {results['report_path']}")
    print(f"Gráficas: {results['figure_dir']}")


if __name__ == "__main__":
    main()
