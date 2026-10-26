"""Persian Markdown reporting and scientific figures for mitdb_pvc_v1.

The report describes saved artifacts only: it never reruns a model or silently
converts missing/failed observations into successful predictions.
"""
from __future__ import annotations

import csv
from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import PaperConfig


LABELS = ("LOW_PVC", "HIGH_PVC")
HOS_FEATURES = ("P1", "P2", "H1", "H2", "H3", "f1m", "f2m")
MODEL_NAMES = {
    "majority": "کلاس اکثریت",
    "dummy": "کلاس اکثریت",
    "baseline": "ویژگی‌های ساده",
    "hos": "هفت ویژگی HOS",
    "combined": "ویژگی‌های ترکیبی",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig") as stream:
        value = json.load(stream)
    return value if isinstance(value, dict) else {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _number(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _pct(value: Any) -> str:
    value = _number(value)
    return "ثبت نشده" if value is None else f"{100 * value:.1f}%"


def _count(value: Any) -> str:
    value = _number(value)
    return "ثبت نشده" if value is None else str(int(value))


def _safe_text(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _setting(value: Any) -> str:
    if isinstance(value, bool):
        return "فعال" if value else "غیرفعال"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return "ثبت نشده" if value is None else str(value)


def _successful(row: dict[str, Any]) -> bool:
    return str(row.get("status", "")).lower() in {"ok", "success", "complete"}


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    result = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    result.extend("| " + " | ".join(_safe_text(cell) for cell in row) + " |" for row in rows)
    return result


def _ci_text(result: dict[str, Any], metric: str) -> str:
    interval = result.get("ci", {}).get(metric)
    if not isinstance(interval, (list, tuple)) or len(interval) != 2:
        return "ثبت نشده"
    if _number(interval[0]) is None or _number(interval[1]) is None:
        return "ثبت نشده"
    return f"{_pct(interval[0])} تا {_pct(interval[1])}"


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_figure(fig, path: Path) -> None:
    plt = _pyplot()
    fig.savefig(path, dpi=170, bbox_inches="tight", metadata={"Software": "hos-hrv-reproduction"})
    plt.close(fig)


def _placeholder(path: Path, title: str, detail: str) -> None:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7, 2.4))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, detail, transform=ax.transAxes, ha="center", va="center", wrap=True)
    _save_figure(fig, path)


def _feature_plot(rows: list[dict[str, str]], path: Path) -> None:
    plt = _pyplot()
    fig, axes = plt.subplots(2, 4, figsize=(13, 6))
    found = False
    for ax, feature in zip(axes.flat, HOS_FEATURES):
        values = []
        for label in LABELS:
            values.append([
                _number(row.get(feature)) for row in rows
                if row.get("label") == label and _successful(row) and _number(row.get(feature)) is not None
            ])
        if any(values):
            found = True
            ax.boxplot([v if v else [np.nan] for v in values], positions=[1, 2], showfliers=False)
            ax.set_xticks([1, 2], LABELS)
            for position, items in enumerate(values, 1):
                if items:
                    offsets = np.linspace(-0.09, 0.09, len(items))
                    ax.scatter(position + offsets, items, s=13, alpha=0.65)
        else:
            ax.text(0.5, 0.5, "No valid values", transform=ax.transAxes, ha="center")
            ax.set_xticks([1, 2], LABELS)
        ax.set_title(feature)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=0.25)
    axes.flat[-1].axis("off")
    fig.suptitle("Reference HOS features by segment label" if found else "HOS features: no successful extraction")
    fig.tight_layout()
    _save_figure(fig, path)


def _performance_plot(results: list[dict[str, Any]], path: Path) -> None:
    usable = [r for r in results if _number(r.get("metrics", {}).get("balanced_accuracy")) is not None]
    if not usable:
        _placeholder(path, "Classification performance", "No completed out-of-fold evaluation is available.")
        return
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(max(7, 1.05 * len(usable)), 4.5))
    positions = np.arange(len(usable))
    values = [float(r["metrics"]["balanced_accuracy"]) for r in usable]
    ax.bar(positions, values, color=["#3575a5" if r.get("source") == "reference" else "#d98e32" for r in usable])
    for index, result in enumerate(usable):
        interval = result.get("ci", {}).get("balanced_accuracy", [])
        if isinstance(interval, (list, tuple)) and len(interval) == 2:
            lo, hi = _number(interval[0]), _number(interval[1])
            if lo is not None and hi is not None:
                ax.vlines(index, lo, hi, color="black", linewidth=1.4)
                ax.hlines([lo, hi], index - 0.08, index + 0.08, color="black", linewidth=1.4)
    ax.set_xticks(positions, [f"{r.get('source', '?')}\n{r.get('model', '?')}" for r in usable], rotation=20, ha="right")
    ax.axhline(0.5, color="gray", linestyle="--", label="Chance balanced accuracy")
    ax.set(ylabel="Out-of-fold balanced accuracy", ylim=(0, 1.05), title="Grouped nested CV; subject-bootstrap 95% intervals")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save_figure(fig, path)


def _confusion_plot(results: list[dict[str, Any]], path: Path) -> None:
    usable = []
    for result in results:
        confusion = result.get("metrics", {}).get("confusion")
        try:
            matrix = np.asarray(confusion, dtype=float)
        except (TypeError, ValueError):
            continue
        if matrix.shape == (2, 2):
            matrix = np.column_stack((matrix, np.zeros(2)))
        if matrix.shape == (2, 3) and np.all(np.isfinite(matrix)):
            usable.append((result, matrix))
    if not usable:
        _placeholder(path, "Confusion matrices", "No confusion matrix is available; failures are not converted to predictions.")
        return
    plt = _pyplot()
    cols = min(4, len(usable))
    rows = int(np.ceil(len(usable) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.1 * cols, 3.1 * rows), squeeze=False)
    for ax, (result, matrix) in zip(axes.flat, usable):
        ax.imshow(matrix, cmap="Blues", vmin=0)
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, f"{value:g}", ha="center", va="center", color="black")
        ax.set_xticks(range(3), ["LOW", "HIGH", "FAIL"], fontsize=8)
        ax.set_yticks(range(2), ["LOW_PVC", "HIGH_PVC"], fontsize=8)
        ax.set(xlabel="Predicted", ylabel="Reference", title=f"{result.get('source', '?')} / {result.get('model', '?')}")
    for ax in list(axes.flat)[len(usable):]:
        ax.axis("off")
    fig.tight_layout()
    _save_figure(fig, path)


def _example_plots(run_dir: Path, figures_dir: Path) -> list[tuple[str, str]]:
    paths = sorted((run_dir / "examples").glob("*.npz"))[:2]
    if not paths:
        return []
    plt = _pyplot()
    examples = []
    for path in paths:
        try:
            with np.load(path, allow_pickle=False) as archive:
                examples.append((path.stem, {key: archive[key] for key in archive.files}))
        except (OSError, ValueError):
            continue
    if not examples:
        return []
    fig, axes = plt.subplots(len(examples), 2, figsize=(13, 3.3 * len(examples)), squeeze=False)
    for (name, values), (ecg_ax, hrv_ax) in zip(examples, axes):
        times = np.asarray(values.get("times_s", []), dtype=float).ravel()
        ecg = np.asarray(values.get("ecg", []), dtype=float).ravel()
        if times.size and times.size == ecg.size:
            # A short visible interval lets R-peak markers be checked at beat scale.
            start = float(times[0])
            visible = (times >= start) & (times <= start + 10)
            ecg_ax.plot(times[visible], ecg[visible], color="#333333", lw=0.8, label="ECG")
            for key, marker, color, label in (("reference_peaks_s", "o", "#1f77b4", "Reference R"), ("detector_peaks_s", "x", "#e67e22", "Detected R")):
                peaks = np.asarray(values.get(key, []), dtype=float).ravel()
                peaks = peaks[(peaks >= start) & (peaks <= start + 10)]
                if peaks.size:
                    ecg_ax.scatter(peaks, np.interp(peaks, times, ecg), s=25, marker=marker, color=color, label=label)
            ecg_ax.legend(fontsize=7)
        else:
            ecg_ax.text(0.5, 0.5, "ECG example not available", transform=ecg_ax.transAxes, ha="center")
        ecg_ax.set(title=f"{name}: first 10 s", xlabel="Time (s)", ylabel="ECG (mV)")
        hrv_times = np.asarray(values.get("hrv_times_s", []), dtype=float).ravel()
        hrv = np.asarray(values.get("hrv", []), dtype=float).ravel()
        if hrv_times.size and hrv_times.size == hrv.size:
            hrv_ax.plot(hrv_times, hrv, lw=0.7, color="#3575a5")
        else:
            hrv_ax.text(0.5, 0.5, "HR series not available", transform=hrv_ax.transAxes, ha="center")
        hrv_ax.set(title=f"{name}: HR series", xlabel="Time (s)", ylabel="Heart rate (bpm)")
        for ax in (ecg_ax, hrv_ax):
            ax.grid(alpha=0.2)
    fig.tight_layout()
    _save_figure(fig, figures_dir / "signal_examples.png")

    fig, axes = plt.subplots(1, len(examples), figsize=(6 * len(examples), 4), squeeze=False)
    for ax, (name, values) in zip(axes.flat, examples):
        magnitude = np.asarray(values.get("bispectrum_magnitude", []), dtype=float)
        frequencies = np.asarray(values.get("f_axis", []), dtype=float).ravel()
        if magnitude.ndim == 2 and magnitude.size and frequencies.size:
            valid = np.isfinite(magnitude) & (magnitude > 0)
            logs = np.full(magnitude.shape, np.nan)
            logs[valid] = np.log10(magnitude[valid])
            im = ax.imshow(logs, origin="lower", aspect="equal", extent=(frequencies[0], frequencies[-1], frequencies[0], frequencies[-1]), cmap="viridis")
            fig.colorbar(im, ax=ax, label="log10 |B| (estimator units)", shrink=0.8)
        else:
            ax.text(0.5, 0.5, "Bispectrum not available", transform=ax.transAxes, ha="center")
        ax.set(title=name, xlabel="f1 (Hz)", ylabel="f2 (Hz)")
    fig.tight_layout()
    _save_figure(fig, figures_dir / "bispectrum_examples.png")
    return [("نمونهٔ سیگنال، قله‌های R و سری ضربان", "signal_examples.png"), ("لگاریتم دامنهٔ بای‌اسپکتروم نمونه‌ها", "bispectrum_examples.png")]


def report_mitdb(run_dir: str | Path) -> Path:
    """Create report_fa.md and figures/ from a prepared/extracted/evaluated run.

    Missing optional inputs are explicitly described in the report. Malformed
    JSON/CSV still raises an error rather than disguising a corrupted run.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    audit = _read_json(run_dir / "audit.json")
    evaluation = _read_json(run_dir / "evaluation.json")
    protocol_settings = _read_json(run_dir / "protocol.json")
    extraction = _read_json(run_dir / "extraction.json")
    paper_config = _read_json(run_dir / "paper_config.json") or extraction.get("paper_config", {})
    cohort = _read_csv(run_dir / "cohort.csv")
    manifest = _read_csv(run_dir / "manifest.csv")
    features = {source: _read_csv(run_dir / f"features_{source}.csv") for source in ("reference", "detector")}
    qrs = _read_csv(run_dir / "qrs_metrics.csv")
    results = [r for r in evaluation.get("results", []) if isinstance(r, dict)]
    report_protocol = protocol_settings or evaluation.get("protocol_settings", {})
    if not protocol_settings and isinstance(report_protocol, dict):
        protocol_settings = report_protocol

    _feature_plot(features["reference"], figures_dir / "hos_features.png")
    _performance_plot(results, figures_dir / "performance.png")
    _confusion_plot(results, figures_dir / "confusion.png")
    example_figures = _example_plots(run_dir, figures_dir)

    protocol = protocol_settings.get("protocol_id", audit.get("protocol_id", evaluation.get("protocol", "mitdb_pvc_v1")))
    low_limit = _pct(protocol_settings.get("low_pvc_max", 0.01))
    high_limit = _pct(protocol_settings.get("high_pvc_min", 0.05))
    segment_minutes = float(protocol_settings.get("segment_seconds", 900)) / 60
    expected_config = PaperConfig().to_dict()
    actual_config = {**expected_config, **paper_config}
    config_origin = "مقادیر ثبت‌شده در paper_config.json یا extraction.json" if paper_config else "پیش‌فرض مورد انتظار؛ فایل تنظیمات استخراج موجود نیست و اجرای آن تأیید نمی‌شود"
    lines = [
        "# گزارش طبقه‌بندی بار PVC با ویژگی‌های HOS",
        "",
        f"پروتکل: `{_safe_text(protocol)}`. این گزارش از فایل‌های همین اجرا ساخته شده است؛ اجرای گزارش‌ساز مدل را آموزش نمی‌دهد.",
        "",
        "## پرسش و تعریف خروجی",
        "",
        f"هدف، بررسی تفکیک قطعات {segment_minutes:g} دقیقه‌ای با بار کم و زیاد ضربان زودرس بطنی در افراد خارج از آموزش است. این آزمایش تشخیص پنج بیماری مقالهٔ اصلی یا تشخیص بالینی نیست.",
        "",
        *(_table(["برچسب", "تعریف عملیاتی"], [["`LOW_PVC`", f"نسبت ضربان‌های V به همهٔ ضربان‌های معتبر حداکثر {low_limit}"], ["`HIGH_PVC`", f"نسبت ضربان‌های V به همهٔ ضربان‌های معتبر حداقل {high_limit}"]])),
        "",
        f"قطعات با بار بیشتر از {low_limit} و کمتر از {high_limit} حذف می‌شوند. تعداد و درصد PVC فقط برای ساخت برچسب است و ورودی مدل نیست. LOW_PVC به معنی فرد سالم نیست؛ این آستانه‌ها معیار تشخیص پزشکی نیستند.",
        "",
        "## داده و ممیزی",
        "",
        "دادهٔ ورودی: [MIT-BIH Arrhythmia Database 1.0.0](https://physionet.org/content/mitdb/1.0.0/). فایل‌های خام در مخزن کد قرار نمی‌گیرند. هش و نسخهٔ داده، همراه فایل‌های آماده‌سازی اجرا نگهداری می‌شود.",
        "",
    ]
    if protocol != "mitdb_pvc_v1":
        lines[4:4] = ["**این اجرا یک آزمایش تغییریافته یا آزمایشی است و پروتکل اصلی mitdb_pvc_v1 محسوب نمی‌شود. تنظیمات و نتایج آن باید جداگانه تفسیر شوند.**", ""]
    if evaluation.get("protocol_deviations"):
        deviations = evaluation["protocol_deviations"]
        if isinstance(deviations, dict):
            deviations = [f"{key}: {value}" for key, value in deviations.items()]
        elif not isinstance(deviations, list):
            deviations = [str(deviations)]
        lines.extend(["تفاوت‌های ثبت‌شده با تنظیمات اصلی:", "", *[f"- {_safe_text(value)}" for value in deviations], ""])
    n_included = audit.get("included_segments", len(manifest))
    n_candidates = audit.get("candidate_segments", len(cohort))
    n_excluded = audit.get("excluded_segments", max(0, int(n_candidates or 0) - int(n_included or 0)))
    lines.extend(_table(["شاخص", "تعداد"], [["قطعات بررسی‌شده", _count(n_candidates)], ["قطعات پذیرفته‌شده", _count(n_included)], ["قطعات حذف‌شده", _count(n_excluded)], ["افراد پذیرفته‌شده", _count(audit.get("n_subjects", len({r.get('subject_id') for r in manifest if r.get('subject_id')})))]]))
    class_counts = audit.get("class_counts", {})
    lines.extend(["", *(_table(["کلاس", "قطعه", "فرد"], [[label, _count(class_counts.get(label, {}).get("segments")), _count(class_counts.get(label, {}).get("subjects"))] for label in LABELS])), ""])
    exclusions = audit.get("exclusion_counts", {})
    if exclusions:
        lines.extend(["علت‌های حذف ممکن است هم‌زمان برای یک قطعه ثبت شده باشند:", "", *(_table(["علت ثبت‌شده", "تعداد"], [[key, value] for key, value in sorted(exclusions.items())])), ""])
    annotation_flags = Counter()
    for row in cohort:
        if row.get("quality_flags"):
            flags = json.loads(row["quality_flags"])
            if isinstance(flags, dict):
                annotation_flags.update({str(key): int(value) for key, value in flags.items()})
    if annotation_flags:
        lines.extend(["نشانه‌های غیرضربان در حاشیه‌نویسی قطعات بررسی‌شده در جدول زیر جمع شده‌اند. همهٔ این کدها الزاماً نشانهٔ کیفیت پایین نیستند؛ برای نمونه، کد تغییر ریتم هم غیرضربان است. وجود آن‌ها به‌تنهایی قاعدهٔ حذف جدید ایجاد نمی‌کند و جزئیات هر قطعه در cohort.csv باقی می‌ماند.", "", *(_table(["کد حاشیه‌نویسی", "تعداد"], [[f"`{key}`", value] for key, value in sorted(annotation_flags.items())])), ""])
    if not audit:
        lines.extend(["فایل audit.json موجود نیست؛ شمارش‌های جایگزین از فایل‌های CSV خوانده شده‌اند و ممیزی کامل در دسترس نیست.", ""])
    lines.extend([
        f"رکوردهای کنارگذاشته‌شده: {_setting(protocol_settings.get('excluded_records', ['102', '104', '107', '217']))}. رکوردهای ۲۰۱ و ۲۰۲ یک فرد محسوب می‌شوند. قطعات بدون هم‌پوشانی از کانال MLII بررسی می‌شوند. قواعد پذیرش شامل سهم حداقل {_pct(protocol_settings.get('min_beat_purity', 0.99))} ضربان‌های مجاز و سهم حداقل {_pct(protocol_settings.get('min_rhythm_coverage', 0.95))} ریتم‌های مجاز است. جزئیات علت حذف در cohort.csv ثبت می‌شود.",
        "",
        "## استخراج ویژگی و پوشش پردازش",
        "",
        "مسیر reference از زمان ضربان‌های حاشیه‌نویسی متخصصان استفاده می‌کند و آزمایش اصلیِ ویژگی‌ها است. مسیر detector از ECG خام، بازنمونه‌برداری، فیلتر و آشکارساز R استفاده می‌کند و خطای زنجیرهٔ کاربردی را هم شامل می‌شود. آشکارسازی روی رکورد کامل انجام می‌شود.",
        "",
        f"در هر مسیر RR برحسب ثانیه و HR=60/RR ساخته و بدون برون‌یابی به سری یکنواخت تبدیل می‌شود. ضربان نابجای واقعی حذف نمی‌شود. بای‌اسپکتروم با میانگین‌گیری بلوک‌ها محاسبه می‌شود؛ حداقل {_count(protocol_settings.get('min_blocks', 5))} بلوک معتبر برای هر قطعه لازم است. یک قطعه فقط یک بردار ویژگی می‌دهد و بلوک‌های داخلی نمونه‌های مستقل نیستند.",
        "",
        "ویژگی‌های HOS عبارت‌اند از P1، P2، H1، H2، H3، f1m و f2m. مجموعهٔ ساده شامل mean_rr، sd_rr، rmssd_rr، prr50، توان لگاریتمی VLF/LF/HF و آنتروپی طیف توان است. این شاخص‌ها از فاصلهٔ تمام ضربان‌ها ساخته می‌شوند و نباید به‌عنوان HRV صرفاً مبتنی بر ضربان‌های نرمال یا شاخص قطعی عملکرد دستگاه عصبی خودمختار تفسیر شوند.",
        "",
    ])
    lines.extend([f"تنظیمات پردازش زیر {config_origin} هستند. مواردی مانند پنجره، پایهٔ لگاریتم و مقیاس FFT از ابهام‌های پیاده‌سازی مقاله‌اند و انتخاب این آزمایش محسوب می‌شوند.", ""])
    config_rows = [
        ["نرخ هدف ECG (Hz)", actual_config.get("ecg_target_fs")],
        ["نرخ سری ضربان (Hz)", actual_config.get("hrv_fs")],
        ["اندازهٔ بلوک / NFFT", actual_config.get("block_size")],
        ["هم‌پوشانی نمونه‌ها", actual_config.get("overlap")],
        ["پنجره", actual_config.get("window")],
        ["درون‌یابی", actual_config.get("interpolation")],
        ["حذف میانگین بلوک", actual_config.get("detrend")],
        ["مقیاس FFT", actual_config.get("fft_scaling")],
        ["پایهٔ لگاریتم", actual_config.get("log_base")],
        ["کف دامنه برای لگاریتم", actual_config.get("log_floor")],
        ["نرمال‌سازی آنتروپی", actual_config.get("normalize_entropy")],
        ["حضور مؤلفهٔ DC", actual_config.get("include_dc")],
        ["مبدأ شاخص H3 و مرکز وزنی", actual_config.get("index_origin")],
        ["پایین‌گذر / بالاگذر (Hz)", f"{actual_config.get('lowpass_hz')} / {actual_config.get('highpass_hz')}"],
        ["مرتبهٔ فیلتر", actual_config.get("filter_order")],
        ["ناچ detector (Hz)", 60],
        ["ضریب کیفیت ناچ", actual_config.get("notch_q")],
        ["عرض فیلتر میانه (s)", actual_config.get("median_seconds")],
    ]
    lines.extend(_table(["پارامتر", "مقدار"], [[name, _setting(value)] for name, value in config_rows]))
    lines.extend(["", "مختصات ویژگی‌های f1m و f2m در پیاده‌سازی حاضر، شاخص خانهٔ FFT با مبدأ ثبت‌شده است؛ محور شکل بای‌اسپکتروم برای خوانایی برحسب Hz نمایش داده می‌شود و با واحد این دو ویژگی یکسان نیست.", "", *(_table(["قاعدهٔ پروتکل", "مقدار"], [
        ["ضربان‌های مجاز", _setting(protocol_settings.get("allowed_beats", ["N", "L", "R", "V"]))],
        ["ریتم‌های مجاز", _setting(protocol_settings.get("allowed_rhythms", ["(N", "(B", "(T", "(SBR"]))],
        ["بیشینهٔ خط صاف پیوسته (s)", _setting(protocol_settings.get("max_flatline_seconds", 2))],
        ["حداقل بلوک در هر مسیر", _count(protocol_settings.get("min_blocks", 5))],
        ["حداقل فرد پذیرفته‌شده در هر کلاس", _count(protocol_settings.get("min_subjects_per_class", 5))],
    ])), ""])
    coverage_rows = []
    for source, rows in features.items():
        valid = sum(_successful(row) for row in rows)
        coverage_rows.append([source, len(rows), valid, len(rows) - valid, _pct(valid / len(rows)) if rows else "ثبت نشده"])
    lines.extend(_table(["مسیر", "قطعات ثبت‌شده", "موفق", "شکست", "پوشش استخراج"], coverage_rows))
    for source, rows in features.items():
        failures = [r for r in rows if not _successful(r)]
        if failures:
            lines.extend(["", f"شکست‌های مسیر `{source}`:", "", *(_table(["قطعه", "پیام خطا"], [[r.get("segment_id", "?"), r.get("error", "ثبت نشده")] for r in failures]))])
    if qrs:
        tp = sum(_number(row.get("tp")) or 0 for row in qrs)
        fp = sum(_number(row.get("fp")) or 0 for row in qrs)
        fn = sum(_number(row.get("fn")) or 0 for row in qrs)
        weighted_mae = sum((_number(row.get("timing_mae_ms")) or 0) * (_number(row.get("tp")) or 0) for row in qrs if _number(row.get("timing_mae_ms")) is not None)
        mae_weight = sum(_number(row.get("tp")) or 0 for row in qrs if _number(row.get("timing_mae_ms")) is not None)
        tolerance_ms = 1000 * float(protocol_settings.get("qrs_tolerance_seconds", 0.15))
        lines.extend(["", f"تطبیق قله‌های آشکارساز با مرجع یک‌به‌یک و با تلورانس {tolerance_ms:g} میلی‌ثانیه است. شاخص‌های زیر از مجموع TP/FP/FN قطعات محاسبه شده‌اند:", "", *(_table(["TP", "FP", "FN", "حساسیت", "ارزش اخباری مثبت", "MAE زمانی"], [[int(tp), int(fp), int(fn), _pct(tp / (tp + fn)) if tp + fn else "ثبت نشده", _pct(tp / (tp + fp)) if tp + fp else "ثبت نشده", f"{weighted_mae / mae_weight:.2f} ms" if mae_weight else "ثبت نشده"]]))])
    lines.extend(["", "## ارزیابی طبقه‌بندی", ""])
    status = evaluation.get("status", "not_available")
    lines.extend([f"وضعیت ثبت‌شده: `{_safe_text(status)}`.", ""])
    if "frozen_extraction_verified" in evaluation:
        verification = "تأیید شده" if evaluation["frozen_extraction_verified"] else "تأیید نشده"
        lines.extend([f"یکپارچگی تنظیمات و هش‌های استخراج در ارزیابی: **{verification}**.", ""])
    if not evaluation:
        lines.extend(["ارزیابی هنوز در این پوشه موجود نیست؛ هیچ عدد عملکرد یا برتری HOS از این گزارش قابل نتیجه‌گیری نیست.", ""])
    elif status != "complete":
        lines.extend(["ارزیابی کامل نشده است. در صورت ناکافی‌بودن افراد هر کلاس، تشکیل‌نشدن تقسیم‌های معتبر یا شکست استخراج، نتیجهٔ نهایی طبقه‌بندی اعلام نمی‌شود. قواعد برای بهترشدن عملکرد تغییر داده نمی‌شوند.", ""])
        if evaluation.get("reason"):
            lines.extend([f"علت ثبت‌شده: {_safe_text(evaluation['reason'])}", ""])
    split_config = evaluation.get("split_config", {})
    lines.extend([
        f"ارزیابی از اعتبارسنجی متقاطع تو‌در‌تو با {_count(split_config.get('outer_splits', 3))} بخش بیرونی و {_count(split_config.get('inner_splits', 3))} بخش درونی در سطح فرد استفاده می‌کند. استانداردسازی و انتخاب C/gamma فقط در آموزش انجام می‌شود. تمام قطعات یک فرد در یک بخش می‌مانند. مدل‌های اکثریت، ویژگی‌های ساده، HOS و ترکیبی از تقسیم‌های یکسان استفاده می‌کنند. بذر پایه {_count(evaluation.get('seed', protocol_settings.get('seed', 20261026)))} است و انتخاب تقسیم به عملکرد وابسته نیست.",
        "",
        f"برای بازه‌های اطمینان ۹۵٪، در صورت موجودبودن، {_count(evaluation.get('bootstrap', {}).get('requested_draws', 2000))} بازنمونه‌گیری در سطح فرد درخواست شده است. بازنمونه‌هایی که هر دو کلاس را ندارند کنار گذاشته می‌شوند؛ تعداد قابل استفاده: {_count(evaluation.get('bootstrap', {}).get('usable_draws'))}. این بازه‌ها عدم‌قطعیت همین پیش‌بینی‌های خارج از آموزش را توصیف می‌کنند. FAIL/ABSTAIN در ماتریس‌ها به معنی شکست زنجیره است؛ در ارزیابی سرتاسری پاسخ نادرست محسوب می‌شود و از مخرج حذف نمی‌شود.",
        "",
    ])
    if results:
        evaluated_cohort = evaluation.get("cohort", {})
        if evaluated_cohort:
            lines.extend(_table(["جمعیت ارزیابی", "تعداد قطعه"], [
                ["قطعات پذیرفته‌شده در آماده‌سازی", _count(evaluated_cohort.get("manifest_count"))],
                ["قطعات با استخراج مرجع موفق؛ شاخص پوشش", _count(evaluated_cohort.get("reference_valid_count"))],
                ["قطعات با استخراج آشکارساز موفق؛ شاخص پوشش", _count(evaluated_cohort.get("detector_success_count"))],
            ]))
            lines.extend(["", "جمعیت اصلی ارزیابی شامل تمام قطعات پذیرفته‌شده پس از کنترل کیفیت داده است. شکست استخراج در هر یک از مسیرهای مرجع و آشکارساز، قطعه را از این جمعیت خارج نمی‌کند: آن مسیر برای قطعه ABSTAIN ثبت می‌کند و در معیارهای سرتاسری پاسخ نادرست محسوب می‌شود. تعداد استخراج‌های موفق فقط شاخص پوشش است و مخرج ارزیابی اصلی را تغییر نمی‌دهد.", ""])
        metric_rows = []
        for result in results:
            metric = result.get("metrics", {})
            metric_rows.append([result.get("source", "?"), MODEL_NAMES.get(result.get("model"), result.get("model", "?")), _pct(metric.get("balanced_accuracy")), _ci_text(result, "balanced_accuracy"), _pct(metric.get("sensitivity")), _pct(metric.get("specificity")), _pct(metric.get("f1_positive")), _pct(metric.get("coverage"))])
        lines.extend(_table(["مسیر", "مدل", "دقت متوازن", "بازهٔ ۹۵٪", "حساسیت HIGH", "ویژگی LOW", "F1 کلاس HIGH", "پوشش"], metric_rows))
        lines.append("")
    comparisons = evaluation.get("comparisons", {})
    for comparison in comparisons.get("hos_minus_baseline", []):
        difference = _number(comparison.get("balanced_accuracy_difference"))
        if difference is not None:
            interval = comparison.get("ci")
            if isinstance(interval, dict):
                interval = interval.get("balanced_accuracy_difference", interval.get("balanced_accuracy"))
            ci = f"؛ بازهٔ اطمینان: {100 * float(interval[0]):.1f} تا {100 * float(interval[1]):.1f} واحد درصد" if isinstance(interval, (list, tuple)) and len(interval) == 2 and all(_number(x) is not None for x in interval) else ""
            lines.extend([f"اختلاف دقت متوازن HOS منهای ویژگی‌های ساده در مسیر `{_safe_text(comparison.get('source', '?'))}`: **{100 * difference:.1f} واحد درصد**{ci}. این عدد باید همراه اندازهٔ نمونه و عدم‌قطعیت تفسیر شود.", ""])
    paired = comparisons.get("paired_sources", [])
    if paired:
        lines.extend(["مقایسهٔ جفت‌شدهٔ دو مسیر فقط روی قطعات موفق مشترک انجام شده است و جایگزین ارزیابی سرتاسریِ شامل شکست‌ها نیست:", "", *(_table(["مدل", "قطعات مشترک", "دقت متوازن مرجع", "دقت متوازن آشکارساز"], [[p.get("model", "?"), p.get("n_segments", "?"), _pct(p.get("reference_metrics", {}).get("balanced_accuracy")), _pct(p.get("detector_metrics", {}).get("balanced_accuracy"))] for p in paired])), ""])
    lines.extend(["## نمودارها", ""])
    for caption, filename in example_figures + [("توزیع هفت ویژگی HOS در مسیر مرجع", "hos_features.png"), ("دقت متوازن و بازهٔ اطمینان مدل‌ها", "performance.png"), ("ماتریس‌های درهم‌ریختگی؛ ستون سوم شامل شکست پردازش", "confusion.png")]:
        lines.extend([f"![{caption}](figures/{filename})", ""])
    lines.extend([
        "## محدودیت‌ها و بازتولیدپذیری",
        "",
        "این یک مطالعهٔ اکتشافی با افراد کم است. قطعات یک فرد مستقل نیستند؛ به همین دلیل تقسیم و بازنمونه‌گیری در سطح فرد انجام می‌شود. حاشیه‌نویسی مرجع با برچسب خروجی تفاوت دارد: زمان ضربان ورودی پردازش است، ولی کد نوع ضربان فقط برای ساخت برچسب قطعه استفاده می‌شود. هیچ شمارش PVC یا شناسهٔ فرد به مدل داده نمی‌شود.",
        "",
        "برتری HOS تضمین نشده است. پایین‌بودن عملکرد یا نبود تفاوت با ویژگی‌های ساده، نتیجهٔ معتبر همین آزمایش است. از این نتایج نمی‌توان تشخیص بالینی، سازوکار فیزیولوژیک قطعی، یا تکرار نتایج مقالهٔ اصلی را نتیجه گرفت.",
        "",
        "مصنوعی‌بودن هر اجرای آزمایشی باید از متادیتای همان اجرا تشخیص داده شود؛ فایل‌های آزمون نرم‌افزار نتیجهٔ بالینی محسوب نمی‌شوند. برای بازتولید، تنظیمات، نسخهٔ کد، هش ورودی‌ها، تقسیم افراد و پیش‌بینی هر قطعه را همراه این گزارش نگه دارید.",
        "",
        "فایل‌های اصلی این اجرا:",
        "",
    ])
    for name in ("audit.json", "cohort.csv", "manifest.csv", "features_reference.csv", "features_detector.csv", "qrs_metrics.csv", "evaluation.json", "predictions.csv", "splits.json", "metadata.json", "protocol.json", "paper_config.json", "extraction.json", "source_files.json", "provenance.json"):
        if (run_dir / name).exists():
            lines.append(f"- [{name}]({name})")
    lines.append("")
    output = run_dir / "report_fa.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    return output
