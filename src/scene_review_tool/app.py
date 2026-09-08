from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
MASK_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
QUALITY_STATUSES = ["unreviewed", "accepted", "needs_correction", "rejected"]
SCENE_SOURCES = ["taxonomy", "dataset_custom", "candidate_new_taxonomy"]


@dataclass(frozen=True)
class Scene:
    level1_id: str
    level1_name: str
    level1_name_en: str
    scene: str
    elements: tuple[str, ...]


@dataclass(frozen=True)
class Sample:
    id: str
    image_path: str
    mask_path: str
    parent_group_id: str
    read_status: str = "ok"


@dataclass
class Review:
    quality_status: str = "unreviewed"
    scene_status: str = "unassigned"
    level1_id: str = ""
    level1_name: str = ""
    primary_level2_scene: str = ""
    secondary_scenes_json: str = "[]"
    scene_source: str = "taxonomy"
    custom_scene_name_en: str = ""
    custom_scene_name_zh: str = ""
    taxonomy_mapping_suggestion: str = ""
    issue_tags_json: str = "[]"
    reviewer_note: str = ""


def now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def normalized_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def stable_sample_id(dataset_name: str, image_path: Path) -> str:
    stat = image_path.stat()
    raw = f"{dataset_name}|{normalized_path(image_path)}|{stat.st_size}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


class Taxonomy:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.version = ""
        self.sha256 = ""
        self.scenes: list[Scene] = []
        self.by_domain: dict[str, list[Scene]] = {}
        self.load()

    def load(self) -> None:
        data = self.path.read_bytes()
        self.sha256 = hashlib.sha256(data).hexdigest()
        obj = json.loads(data.decode("utf-8"))
        self.version = str(obj.get("version", ""))
        scenes: list[Scene] = []
        for domain in obj.get("domains", []):
            for scene in domain.get("scenes", []):
                scenes.append(
                    Scene(
                        level1_id=str(domain.get("id", "")),
                        level1_name=str(domain.get("name_zh", "")),
                        level1_name_en=str(domain.get("name_en", "")),
                        scene=str(scene.get("scene", "")),
                        elements=tuple(scene.get("elements", [])),
                    )
                )
        self.scenes = scenes
        self.by_domain = {}
        for scene in scenes:
            self.by_domain.setdefault(scene.level1_id, []).append(scene)

    @property
    def domains(self) -> list[tuple[str, str, str]]:
        seen: dict[str, tuple[str, str, str]] = {}
        for scene in self.scenes:
            seen[scene.level1_id] = (scene.level1_id, scene.level1_name, scene.level1_name_en)
        return list(seen.values())

    def find_scene(self, name: str) -> Scene | None:
        for scene in self.scenes:
            if scene.scene == name:
                return scene
        return None


class ReviewDatabase:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self.migrate()

    def migrate(self) -> None:
        cur = self.conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                image_root TEXT NOT NULL,
                mask_root TEXT NOT NULL,
                context_root TEXT,
                config_json TEXT NOT NULL,
                taxonomy_version TEXT,
                taxonomy_sha256 TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS samples (
                id TEXT PRIMARY KEY,
                dataset_id INTEGER NOT NULL,
                image_path TEXT NOT NULL,
                mask_path TEXT NOT NULL,
                context_path TEXT,
                parent_group_id TEXT,
                image_width INTEGER,
                image_height INTEGER,
                image_dtype TEXT,
                mask_width INTEGER,
                mask_height INTEGER,
                read_status TEXT,
                content_sha256 TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS reviews (
                sample_id TEXT PRIMARY KEY,
                quality_status TEXT NOT NULL,
                scene_status TEXT NOT NULL,
                level1_id TEXT,
                level1_name TEXT,
                primary_level2_scene TEXT,
                secondary_scenes_json TEXT,
                scene_source TEXT,
                custom_scene_name_en TEXT,
                custom_scene_name_zh TEXT,
                taxonomy_mapping_suggestion TEXT,
                issue_tags_json TEXT,
                reviewer_note TEXT,
                reviewed_at TEXT,
                updated_at TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS dataset_scene_vocab (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id INTEGER NOT NULL,
                scene_source TEXT NOT NULL,
                scene_name_en TEXT NOT NULL,
                scene_name_zh TEXT,
                level1_id TEXT,
                level1_name TEXT,
                mapped_level2_scene TEXT,
                is_pinned INTEGER DEFAULT 0,
                use_count INTEGER DEFAULT 0,
                last_used_at TEXT,
                created_at TEXT NOT NULL,
                UNIQUE(dataset_id, scene_source, scene_name_en)
            )
            """
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def create_dataset(
        self,
        name: str,
        image_root: Path,
        mask_root: Path,
        config: dict[str, Any],
        taxonomy: Taxonomy,
    ) -> int:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO datasets
            (name, image_root, mask_root, context_root, config_json, taxonomy_version, taxonomy_sha256, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name,
                normalized_path(image_root),
                normalized_path(mask_root),
                "",
                json.dumps(config, ensure_ascii=False),
                taxonomy.version,
                taxonomy.sha256,
                now_text(),
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def latest_dataset_id(self) -> int | None:
        row = self.conn.execute("SELECT id FROM datasets ORDER BY id DESC LIMIT 1").fetchone()
        return int(row["id"]) if row else None

    def latest_dataset(self) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM datasets ORDER BY id DESC LIMIT 1").fetchone()

    def add_samples(self, samples: list[Sample], dataset_id: int) -> None:
        rows = [
            (
                s.id,
                dataset_id,
                s.image_path,
                s.mask_path,
                "",
                s.parent_group_id,
                None,
                None,
                "",
                None,
                None,
                s.read_status,
                "",
            )
            for s in samples
        ]
        self.conn.executemany(
            """
            INSERT OR IGNORE INTO samples
            (id, dataset_id, image_path, mask_path, context_path, parent_group_id, image_width, image_height,
             image_dtype, mask_width, mask_height, read_status, content_sha256)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self.conn.commit()

    def samples(self, dataset_id: int, status_filter: str = "all", text_filter: str = "") -> list[tuple[Sample, Review]]:
        sql = """
            SELECT s.*, r.quality_status, r.scene_status, r.level1_id, r.level1_name,
                   r.primary_level2_scene, r.secondary_scenes_json, r.scene_source,
                   r.custom_scene_name_en, r.custom_scene_name_zh, r.taxonomy_mapping_suggestion,
                   r.issue_tags_json, r.reviewer_note
            FROM samples s
            LEFT JOIN reviews r ON r.sample_id = s.id
            WHERE s.dataset_id = ?
        """
        params: list[Any] = [dataset_id]
        if status_filter == "unreviewed":
            sql += " AND COALESCE(r.quality_status, 'unreviewed') = 'unreviewed'"
        elif status_filter == "accepted":
            sql += " AND COALESCE(r.quality_status, 'unreviewed') = 'accepted'"
        elif status_filter == "needs_correction":
            sql += " AND COALESCE(r.quality_status, 'unreviewed') = 'needs_correction'"
        elif status_filter == "rejected":
            sql += " AND COALESCE(r.quality_status, 'unreviewed') = 'rejected'"
        elif status_filter == "unassigned":
            sql += " AND COALESCE(r.scene_status, 'unassigned') = 'unassigned'"
        if text_filter.strip():
            sql += " AND (s.image_path LIKE ? OR s.id LIKE ? OR COALESCE(r.reviewer_note, '') LIKE ?)"
            like = f"%{text_filter.strip()}%"
            params.extend([like, like, like])
        sql += " ORDER BY s.parent_group_id, s.image_path"
        result: list[tuple[Sample, Review]] = []
        for row in self.conn.execute(sql, params).fetchall():
            sample = Sample(
                id=row["id"],
                image_path=row["image_path"],
                mask_path=row["mask_path"],
                parent_group_id=row["parent_group_id"] or "",
                read_status=row["read_status"] or "ok",
            )
            review = Review(
                quality_status=row["quality_status"] or "unreviewed",
                scene_status=row["scene_status"] or "unassigned",
                level1_id=row["level1_id"] or "",
                level1_name=row["level1_name"] or "",
                primary_level2_scene=row["primary_level2_scene"] or "",
                secondary_scenes_json=row["secondary_scenes_json"] or "[]",
                scene_source=row["scene_source"] or "taxonomy",
                custom_scene_name_en=row["custom_scene_name_en"] or "",
                custom_scene_name_zh=row["custom_scene_name_zh"] or "",
                taxonomy_mapping_suggestion=row["taxonomy_mapping_suggestion"] or "",
                issue_tags_json=row["issue_tags_json"] or "[]",
                reviewer_note=row["reviewer_note"] or "",
            )
            result.append((sample, review))
        return result

    def save_review(self, sample_id: str, dataset_id: int, review: Review) -> None:
        review.updated_at = now_text() if hasattr(review, "updated_at") else now_text()
        reviewed_at = now_text() if review.quality_status != "unreviewed" or review.scene_status != "unassigned" else ""
        self.conn.execute(
            """
            INSERT INTO reviews
            (sample_id, quality_status, scene_status, level1_id, level1_name, primary_level2_scene,
             secondary_scenes_json, scene_source, custom_scene_name_en, custom_scene_name_zh,
             taxonomy_mapping_suggestion, issue_tags_json, reviewer_note, reviewed_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(sample_id) DO UPDATE SET
              quality_status=excluded.quality_status,
              scene_status=excluded.scene_status,
              level1_id=excluded.level1_id,
              level1_name=excluded.level1_name,
              primary_level2_scene=excluded.primary_level2_scene,
              secondary_scenes_json=excluded.secondary_scenes_json,
              scene_source=excluded.scene_source,
              custom_scene_name_en=excluded.custom_scene_name_en,
              custom_scene_name_zh=excluded.custom_scene_name_zh,
              taxonomy_mapping_suggestion=excluded.taxonomy_mapping_suggestion,
              issue_tags_json=excluded.issue_tags_json,
              reviewer_note=excluded.reviewer_note,
              reviewed_at=excluded.reviewed_at,
              updated_at=excluded.updated_at
            """,
            (
                sample_id,
                review.quality_status,
                review.scene_status,
                review.level1_id,
                review.level1_name,
                review.primary_level2_scene,
                review.secondary_scenes_json,
                review.scene_source,
                review.custom_scene_name_en,
                review.custom_scene_name_zh,
                review.taxonomy_mapping_suggestion,
                review.issue_tags_json,
                review.reviewer_note,
                reviewed_at,
                now_text(),
            ),
        )
        self._touch_scene_vocab(dataset_id, review)
        self.conn.commit()

    def _touch_scene_vocab(self, dataset_id: int, review: Review) -> None:
        scene_name = review.primary_level2_scene
        scene_zh = review.primary_level2_scene
        mapped = review.primary_level2_scene
        if review.scene_source != "taxonomy":
            scene_name = review.custom_scene_name_en.strip()
            scene_zh = review.custom_scene_name_zh.strip()
            mapped = review.taxonomy_mapping_suggestion.strip()
        if not scene_name:
            return
        self.conn.execute(
            """
            INSERT INTO dataset_scene_vocab
            (dataset_id, scene_source, scene_name_en, scene_name_zh, level1_id, level1_name,
             mapped_level2_scene, is_pinned, use_count, last_used_at, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 0, 1, ?, ?)
            ON CONFLICT(dataset_id, scene_source, scene_name_en) DO UPDATE SET
              scene_name_zh=excluded.scene_name_zh,
              level1_id=excluded.level1_id,
              level1_name=excluded.level1_name,
              mapped_level2_scene=excluded.mapped_level2_scene,
              last_used_at=excluded.last_used_at
            """,
            (
                dataset_id,
                review.scene_source,
                scene_name,
                scene_zh,
                review.level1_id,
                review.level1_name,
                mapped,
                now_text(),
                now_text(),
            ),
        )

    def common_scenes(self, dataset_id: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            """
            WITH assigned AS (
                SELECT
                    COALESCE(scene_source, 'taxonomy') AS scene_source,
                    CASE
                        WHEN COALESCE(scene_source, 'taxonomy') = 'taxonomy'
                        THEN primary_level2_scene
                        ELSE custom_scene_name_en
                    END AS scene_name_en,
                    COUNT(*) AS assigned_count
                FROM reviews
                WHERE sample_id IN (SELECT id FROM samples WHERE dataset_id = ?)
                  AND scene_status IN ('assigned', 'uncertain')
                  AND COALESCE(
                        CASE
                            WHEN COALESCE(scene_source, 'taxonomy') = 'taxonomy'
                            THEN primary_level2_scene
                            ELSE custom_scene_name_en
                        END,
                        ''
                      ) <> ''
                GROUP BY scene_source, scene_name_en
            )
            SELECT v.*, COALESCE(a.assigned_count, 0) AS assigned_count
            FROM dataset_scene_vocab v
            LEFT JOIN assigned a
              ON a.scene_source = v.scene_source
             AND a.scene_name_en = v.scene_name_en
            WHERE v.dataset_id = ?
            ORDER BY v.is_pinned DESC, assigned_count DESC, v.last_used_at DESC, v.scene_name_en
            """,
            (dataset_id, dataset_id),
        ).fetchall()

    def stats(self, dataset_id: int) -> dict[str, Any]:
        total = self.conn.execute("SELECT COUNT(*) AS n FROM samples WHERE dataset_id = ?", (dataset_id,)).fetchone()["n"]
        quality = self.conn.execute(
            """
            SELECT COALESCE(r.quality_status, 'unreviewed') AS status, COUNT(*) AS n
            FROM samples s LEFT JOIN reviews r ON r.sample_id = s.id
            WHERE s.dataset_id = ?
            GROUP BY COALESCE(r.quality_status, 'unreviewed')
            """,
            (dataset_id,),
        ).fetchall()
        scenes = self.conn.execute(
            """
            SELECT COALESCE(NULLIF(r.primary_level2_scene, ''), NULLIF(r.custom_scene_name_en, ''), '_unassigned') AS scene,
                   COALESCE(r.scene_source, 'taxonomy') AS source,
                   COUNT(*) AS n
            FROM samples s LEFT JOIN reviews r ON r.sample_id = s.id
            WHERE s.dataset_id = ?
            GROUP BY scene, source
            ORDER BY n DESC, scene
            """,
            (dataset_id,),
        ).fetchall()
        return {
            "total": total,
            "quality": {row["status"]: row["n"] for row in quality},
            "scenes": [(row["scene"], row["source"], row["n"]) for row in scenes],
        }


def scan_dataset(
    dataset_name: str,
    image_root: Path,
    mask_root: Path,
    mode: str,
    image_dir_name: str,
    mask_dir_name: str,
    split_name: str = "all",
    recursive: bool = True,
) -> tuple[list[Sample], list[str]]:
    samples: list[Sample] = []
    warnings: list[str] = []
    if mode == "voc_segmentation":
        image_dir = image_root / image_dir_name
        mask_dir = mask_root / mask_dir_name
        if not image_dir.is_dir():
            image_dir = image_root
        if not mask_dir.is_dir():
            mask_dir = mask_root

        image_by_stem = {
            path.stem: path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        }
        mask_by_stem = {
            path.stem: path
            for path in mask_dir.iterdir()
            if path.is_file() and path.suffix.lower() in MASK_EXTS
        }

        split = split_name.strip().lower() or "all"
        if split == "all":
            sample_ids = sorted(image_by_stem)
        else:
            split_file = image_root / "ImageSets" / "Segmentation" / f"{split}.txt"
            if not split_file.is_file():
                raise ValueError(f"VOC 划分文件不存在: {split_file}")
            sample_ids = [
                line.strip().split()[0]
                for line in split_file.read_text(encoding="utf-8-sig").splitlines()
                if line.strip()
            ]

        seen_ids: set[str] = set()
        for sample_name in sample_ids:
            if sample_name in seen_ids:
                warnings.append(f"duplicate sample in VOC split: {sample_name}")
                continue
            seen_ids.add(sample_name)
            image_path = image_by_stem.get(sample_name)
            mask_path = mask_by_stem.get(sample_name)
            if image_path is None:
                warnings.append(f"missing image for VOC id: {sample_name}")
                continue
            if mask_path is None:
                warnings.append(f"missing mask for VOC id: {sample_name}")
                continue
            samples.append(
                Sample(
                    id=stable_sample_id(dataset_name, image_path),
                    image_path=normalized_path(image_path),
                    mask_path=normalized_path(mask_path),
                    parent_group_id=split,
                )
            )
    elif mode == "mirrored_subdirectories":
        image_files = [p for p in image_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS and image_dir_name in p.parts]
        for image_path in image_files:
            parts = list(image_path.parts)
            try:
                idx = len(parts) - 1 - parts[::-1].index(image_dir_name)
            except ValueError:
                continue
            parts[idx] = mask_dir_name
            candidate = Path(*parts)
            mask_path = find_mask_with_extensions(candidate, MASK_EXTS)
            if mask_path is None:
                warnings.append(f"missing mask: {image_path}")
                continue
            parent = image_path.parent.parent.name if image_path.parent.parent else ""
            samples.append(
                Sample(
                    id=stable_sample_id(dataset_name, image_path),
                    image_path=normalized_path(image_path),
                    mask_path=normalized_path(mask_path),
                    parent_group_id=parent,
                )
            )
    else:
        image_iter = image_root.rglob("*") if recursive else image_root.iterdir()
        mask_iter = mask_root.rglob("*") if recursive else mask_root.iterdir()
        image_files = sorted(
            path for path in image_iter if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        )
        mask_files = sorted(
            path for path in mask_iter if path.is_file() and path.suffix.lower() in MASK_EXTS
        )

        def pairing_key(path: Path, root: Path) -> str:
            if mode == "relative_path_stem":
                return path.relative_to(root).with_suffix("").as_posix().lower()
            return path.stem.lower()

        images_by_key: dict[str, list[Path]] = {}
        masks_by_key: dict[str, list[Path]] = {}
        for image_path in image_files:
            images_by_key.setdefault(pairing_key(image_path, image_root), []).append(image_path)
        for mask_path in mask_files:
            masks_by_key.setdefault(pairing_key(mask_path, mask_root), []).append(mask_path)

        duplicate_keys = {
            key for key, paths in images_by_key.items() if len(paths) > 1
        } | {
            key for key, paths in masks_by_key.items() if len(paths) > 1
        }
        for key in sorted(duplicate_keys):
            warnings.append(f"ambiguous duplicate pairing key: {key}")

        for image_path in image_files:
            key = pairing_key(image_path, image_root)
            if key in duplicate_keys:
                continue
            mask_candidates = masks_by_key.get(key, [])
            mask_path = mask_candidates[0] if mask_candidates else None
            if mask_path is None:
                warnings.append(f"missing mask: {image_path}")
                continue
            parent = image_path.parent.name
            samples.append(
                Sample(
                    id=stable_sample_id(dataset_name, image_path),
                    image_path=normalized_path(image_path),
                    mask_path=normalized_path(mask_path),
                    parent_group_id=parent,
                )
            )
        for key, paths in masks_by_key.items():
            if key not in images_by_key:
                for mask_path in paths:
                    warnings.append(f"orphan mask: {mask_path}")
    return samples, warnings


def supported_files(root: Path, extensions: set[str], recursive: bool = True) -> list[Path]:
    iterator = root.rglob("*") if recursive else root.iterdir()
    return sorted(path for path in iterator if path.is_file() and path.suffix.lower() in extensions)


def inspect_sample_pairs(samples: list[Sample], limit: int = 24) -> dict[str, Any]:
    if not samples:
        return {"checked": 0, "mask_values": [], "size_mismatches": 0, "read_errors": 0}
    count = min(limit, len(samples))
    if count == 1:
        selected = [samples[0]]
    else:
        indices = {round(index * (len(samples) - 1) / (count - 1)) for index in range(count)}
        selected = [samples[index] for index in sorted(indices)]

    values: set[int] = set()
    size_mismatches = 0
    read_errors = 0
    for sample in selected:
        try:
            with Image.open(sample.image_path) as image, Image.open(sample.mask_path) as mask:
                if image.size != mask.size:
                    size_mismatches += 1
                colors = mask.getcolors(maxcolors=4096)
                if colors is not None:
                    for _count, value in colors:
                        if isinstance(value, int):
                            values.add(value)
                        elif isinstance(value, tuple) and value:
                            values.add(int(value[0]))
        except Exception:
            read_errors += 1
    return {
        "checked": len(selected),
        "mask_values": sorted(values),
        "size_mismatches": size_mismatches,
        "read_errors": read_errors,
    }


def find_mask_with_extensions(path: Path, exts: set[str]) -> Path | None:
    if path.exists():
        return path
    for ext in exts:
        candidate = path.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


def pil_to_qpixmap(image: Image.Image) -> QPixmap:
    rgba = image.convert("RGBA")
    data = rgba.tobytes("raw", "RGBA")
    qimage = QImage(data, rgba.width, rgba.height, rgba.width * 4, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(qimage.copy())


def compose_preview(
    image_path: str,
    mask_path: str,
    show_mask: bool,
    opacity: int,
    foreground_values: set[int] | None = None,
) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    if not show_mask:
        return image
    mask = Image.open(mask_path).convert("L")
    if mask.size != image.size:
        raise ValueError(f"image and mask size mismatch: {image.size} vs {mask.size}")
    overlay = Image.new("RGBA", image.size, (0, 215, 255, 0))
    if foreground_values:
        alpha = mask.point(lambda value: int(opacity * 2.55) if value in foreground_values else 0)
    else:
        alpha = mask.point(lambda value: int(opacity * 2.55) if value > 0 else 0)
    overlay.putalpha(alpha)
    return Image.alpha_composite(image.convert("RGBA"), overlay)


class ImageCanvas(QScrollArea):
    def __init__(self) -> None:
        super().__init__()
        self.label = QLabel("No image")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setMinimumSize(QSize(400, 400))
        self.setWidget(self.label)
        self.setWidgetResizable(True)
        self.original: QPixmap | None = None
        self.zoom = 1.0

    def set_pixmap(self, pixmap: QPixmap) -> None:
        self.original = pixmap
        self.zoom = 1.0
        self._apply_zoom()

    def set_error(self, text: str) -> None:
        self.original = None
        self.label.setPixmap(QPixmap())
        self.label.setText(text)

    def wheelEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self.original is None:
            return super().wheelEvent(event)
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = 1.15 if event.angleDelta().y() > 0 else 0.87
            self.zoom = min(8.0, max(0.1, self.zoom * delta))
            self._apply_zoom()
            event.accept()
            return
        super().wheelEvent(event)

    def fit(self) -> None:
        self.zoom = 1.0
        self._apply_zoom()

    def _apply_zoom(self) -> None:
        if self.original is None:
            return
        if self.zoom == 1.0:
            area = self.viewport().size()
            scaled = self.original.scaled(area, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        else:
            scaled = self.original.scaled(
                int(self.original.width() * self.zoom),
                int(self.original.height() * self.zoom),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        self.label.setPixmap(scaled)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Scene Review Tool")
        self.resize(1500, 900)
        self.taxonomy_path = Path("D:/硕士毕业论文/ppt相关/geo_scene_taxonomy_1.0.json")
        self.taxonomy = Taxonomy(self.taxonomy_path)
        self.db: ReviewDatabase | None = None
        self.dataset_id: int | None = None
        self.mask_foreground_values: set[int] | None = {1}
        self.items: list[tuple[Sample, Review]] = []
        self.current_index = -1
        self.loading = False

        self.stack = QStackedWidget()
        self.project_page = self._build_project_page()
        self.review_page = self._build_review_page()
        self.stack.addWidget(self.project_page)
        self.stack.addWidget(self.review_page)
        self.setCentralWidget(self.stack)
        self._apply_style()

    def _build_project_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        title = QLabel("遥感图像-mask 场景审查工具")
        title.setObjectName("title")
        subtitle = QLabel("导入数据集后逐图查看原图与 mask 叠加，人工确认质量和场景，所有结果自动保存。")
        subtitle.setObjectName("subtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        form_box = QGroupBox("项目导入")
        form = QFormLayout(form_box)
        greenland_root = "D:/硕士毕业论文/相关图文数据集/自有数据集/greenland_3968x3968_test512"
        self.dataset_name_edit = QLineEdit("Greenland-512")
        self.workspace_edit = QLineEdit("D:/硕士毕业论文/scene_review_tool/workspaces/Greenland-512")
        self.taxonomy_edit = QLineEdit(str(self.taxonomy_path))

        self.import_type_combo = QComboBox()
        self.import_type_combo.addItem("通用图像-Mask文件夹", "generic")
        self.import_type_combo.addItem("VOC语义分割数据集", "voc")

        self.import_options_stack = QStackedWidget()
        generic_page = QWidget()
        generic_form = QFormLayout(generic_page)
        generic_form.setContentsMargins(0, 4, 0, 4)
        self.image_root_edit = QLineEdit(f"{greenland_root}/JPEGImages")
        self.mask_root_edit = QLineEdit(f"{greenland_root}/SegmentationClass")
        self.generic_pairing_combo = QComboBox()
        self.generic_pairing_combo.addItem("同名文件（忽略扩展名）", "same_stem")
        self.generic_pairing_combo.addItem("相对路径与文件名均相同", "relative_path_stem")
        self.recursive_checkbox = QCheckBox("扫描子文件夹")
        self.recursive_checkbox.setChecked(True)
        generic_form.addRow("图像文件夹", self._path_row(self.image_root_edit, True))
        generic_form.addRow("Mask文件夹", self._path_row(self.mask_root_edit, True))
        generic_form.addRow("配对规则", self.generic_pairing_combo)
        generic_form.addRow("", self.recursive_checkbox)
        self.import_options_stack.addWidget(generic_page)

        voc_page = QWidget()
        voc_form = QFormLayout(voc_page)
        voc_form.setContentsMargins(0, 4, 0, 4)
        self.voc_root_edit = QLineEdit(greenland_root)
        self.image_dir_name_edit = QLineEdit("JPEGImages")
        self.mask_dir_name_edit = QLineEdit("SegmentationClass")
        self.split_combo = QComboBox()
        self.split_combo.setEditable(True)
        self.split_combo.addItems(["all", "train", "val", "test"])
        voc_form.addRow("VOC根目录", self._path_row(self.voc_root_edit, True))
        voc_form.addRow("数据划分", self.split_combo)
        voc_form.addRow("图像目录名", self.image_dir_name_edit)
        voc_form.addRow("Mask目录名", self.mask_dir_name_edit)
        self.import_options_stack.addWidget(voc_page)
        self.import_type_combo.currentIndexChanged.connect(self.import_options_stack.setCurrentIndex)

        self.mask_values_edit = QLineEdit("1")
        self.mask_values_edit.setPlaceholderText("多个标签用英文逗号分隔，例如 1,2")
        self.mask_name_edit = QLineEdit("green space")
        form.addRow("数据集名称", self.dataset_name_edit)
        form.addRow("导入方式", self.import_type_combo)
        form.addRow(self.import_options_stack)
        form.addRow("工作区", self._path_row(self.workspace_edit, True))
        form.addRow("场景体系 JSON", self._path_row(self.taxonomy_edit, False))
        form.addRow("Mask 前景标签值", self.mask_values_edit)
        form.addRow("Mask 前景名称", self.mask_name_edit)
        layout.addWidget(form_box)

        action_row = QHBoxLayout()
        scan_btn = QPushButton("扫描预览")
        scan_btn.clicked.connect(self.preview_import)
        create_btn = QPushButton("导入并开始审查")
        create_btn.clicked.connect(self.create_or_open_project)
        open_btn = QPushButton("打开已有工作区")
        open_btn.clicked.connect(self.open_existing_workspace)
        action_row.addWidget(scan_btn)
        action_row.addWidget(create_btn)
        action_row.addWidget(open_btn)
        action_row.addStretch()
        layout.addLayout(action_row)

        preview_row = QHBoxLayout()
        self.project_message = QLabel("请选择导入方式和数据路径，然后先执行扫描预览。")
        self.project_message.setWordWrap(True)
        self.project_message.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.preview_image_label = QLabel("尚未生成配对预览")
        self.preview_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_image_label.setMinimumSize(360, 220)
        self.preview_image_label.setMaximumSize(520, 300)
        self.preview_image_label.setFrameShape(QFrame.Shape.StyledPanel)
        preview_row.addWidget(self.project_message, 1)
        preview_row.addWidget(self.preview_image_label, 1)
        layout.addLayout(preview_row)
        layout.addStretch()
        return page

    def _path_row(self, edit: QLineEdit, directory: bool) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        btn = QPushButton("选择")
        btn.clicked.connect(lambda: self._browse_path(edit, directory))
        layout.addWidget(edit)
        layout.addWidget(btn)
        return row

    def _browse_path(self, edit: QLineEdit, directory: bool) -> None:
        if directory:
            value = QFileDialog.getExistingDirectory(self, "选择目录", edit.text())
        else:
            value, _ = QFileDialog.getOpenFileName(self, "选择文件", edit.text(), "JSON (*.json);;All files (*.*)")
        if value:
            edit.setText(value)

    def _build_review_page(self) -> QWidget:
        page = QWidget()
        root = QVBoxLayout(page)

        toolbar = QHBoxLayout()
        self.dataset_label = QLabel("")
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("搜索文件名 / 样本 ID / 备注")
        self.search_edit.returnPressed.connect(self.reload_samples)
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["all", "unreviewed", "accepted", "needs_correction", "rejected", "unassigned"])
        self.filter_combo.currentTextChanged.connect(self.reload_samples)
        stats_btn = QPushButton("导出统计")
        stats_btn.clicked.connect(self.export_stats)
        export_btn = QPushButton("按场景导出")
        export_btn.clicked.connect(self.export_by_scene)
        back_btn = QPushButton("项目页")
        back_btn.clicked.connect(lambda: self.stack.setCurrentWidget(self.project_page))
        toolbar.addWidget(self.dataset_label)
        toolbar.addWidget(self.search_edit, 2)
        toolbar.addWidget(self.filter_combo)
        toolbar.addWidget(stats_btn)
        toolbar.addWidget(export_btn)
        toolbar.addWidget(back_btn)
        root.addLayout(toolbar)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.sample_list = QListWidget()
        self.sample_list.currentRowChanged.connect(self.on_sample_selected)
        splitter.addWidget(self.sample_list)

        center = QWidget()
        center_layout = QVBoxLayout(center)
        view_bar = QHBoxLayout()
        self.mask_checkbox = QCheckBox("显示 mask")
        self.mask_checkbox.setChecked(True)
        self.mask_checkbox.stateChanged.connect(self.refresh_image)
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(5, 95)
        self.opacity_slider.setValue(45)
        self.opacity_slider.valueChanged.connect(self.refresh_image)
        fit_btn = QPushButton("适合窗口")
        fit_btn.clicked.connect(lambda: self.image_canvas.fit())
        view_bar.addWidget(self.mask_checkbox)
        view_bar.addWidget(QLabel("透明度"))
        view_bar.addWidget(self.opacity_slider)
        view_bar.addWidget(fit_btn)
        center_layout.addLayout(view_bar)
        self.image_canvas = ImageCanvas()
        center_layout.addWidget(self.image_canvas, 1)
        self.path_label = QLabel("")
        self.path_label.setWordWrap(True)
        center_layout.addWidget(self.path_label)
        splitter.addWidget(center)

        panel = self._build_review_panel()
        splitter.addWidget(panel)
        splitter.setSizes([300, 820, 380])
        root.addWidget(splitter, 1)

        bottom = QHBoxLayout()
        prev_btn = QPushButton("上一张")
        next_btn = QPushButton("下一张")
        save_btn = QPushButton("保存")
        next_unreviewed_btn = QPushButton("下一张未审")
        prev_btn.clicked.connect(self.prev_sample)
        next_btn.clicked.connect(self.next_sample)
        save_btn.clicked.connect(self.save_current_review)
        next_unreviewed_btn.clicked.connect(self.next_unreviewed)
        self.status_label = QLabel("")
        bottom.addWidget(prev_btn)
        bottom.addWidget(next_btn)
        bottom.addWidget(next_unreviewed_btn)
        bottom.addWidget(save_btn)
        bottom.addWidget(self.status_label, 1)
        root.addLayout(bottom)
        return page

    def _build_review_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        quality_box = QGroupBox("质量状态")
        q_layout = QVBoxLayout(quality_box)
        self.quality_group = QButtonGroup(self)
        for status in QUALITY_STATUSES:
            radio = QRadioButton(status)
            self.quality_group.addButton(radio)
            q_layout.addWidget(radio)
            if status == "unreviewed":
                radio.setChecked(True)
        self.quality_group.buttonClicked.connect(self.save_quality_status_only)
        layout.addWidget(quality_box)

        common_box = QGroupBox("当前数据集常用场景")
        common_layout = QVBoxLayout(common_box)
        self.common_scene_list = QListWidget()
        self.common_scene_list.itemClicked.connect(self.apply_common_scene)
        common_layout.addWidget(self.common_scene_list)
        layout.addWidget(common_box, 2)

        scene_box = QGroupBox("场景选择")
        scene_layout = QFormLayout(scene_box)
        self.scene_source_combo = QComboBox()
        self.scene_source_combo.addItems(SCENE_SOURCES)
        self.scene_source_combo.currentTextChanged.connect(self.on_scene_source_changed)
        self.domain_combo = QComboBox()
        for domain_id, name_zh, name_en in self.taxonomy.domains:
            self.domain_combo.addItem(f"{domain_id} {name_zh} / {name_en}", domain_id)
        self.domain_combo.currentIndexChanged.connect(self.reload_scene_combo)
        self.scene_combo = QComboBox()
        self.custom_en_edit = QLineEdit()
        self.custom_en_edit.setPlaceholderText("例如 urban waterfront")
        self.custom_zh_edit = QLineEdit()
        self.custom_zh_edit.setPlaceholderText("中文说明")
        self.mapping_edit = QLineEdit()
        self.mapping_edit.setPlaceholderText("可选：映射到正式场景，如 river")
        self.scene_status_combo = QComboBox()
        self.scene_status_combo.addItems(["unassigned", "assigned", "uncertain"])
        scene_layout.addRow("来源", self.scene_source_combo)
        scene_layout.addRow("一级", self.domain_combo)
        scene_layout.addRow("二级", self.scene_combo)
        scene_layout.addRow("自定义英文", self.custom_en_edit)
        scene_layout.addRow("自定义中文", self.custom_zh_edit)
        scene_layout.addRow("映射建议", self.mapping_edit)
        scene_layout.addRow("场景状态", self.scene_status_combo)
        scene_actions = QWidget()
        scene_actions_layout = QHBoxLayout(scene_actions)
        scene_actions_layout.setContentsMargins(0, 0, 0, 0)
        confirm_scene_btn = QPushButton("确认场景")
        confirm_scene_btn.clicked.connect(self.confirm_scene_selection)
        reset_taxonomy_btn = QPushButton("改用正式体系")
        reset_taxonomy_btn.clicked.connect(self.reset_to_taxonomy_scene)
        clear_scene_btn = QPushButton("清空")
        clear_scene_btn.clicked.connect(self.clear_scene_selection)
        scene_actions_layout.addWidget(confirm_scene_btn)
        scene_actions_layout.addWidget(reset_taxonomy_btn)
        scene_actions_layout.addWidget(clear_scene_btn)
        scene_layout.addRow("", scene_actions)
        self.scene_hint_label = QLabel("点击常用场景只填入候选，确认后保存。")
        self.scene_hint_label.setObjectName("hint")
        self.scene_hint_label.setWordWrap(True)
        scene_layout.addRow("", self.scene_hint_label)
        layout.addWidget(scene_box)

        note_box = QGroupBox("备注")
        note_layout = QVBoxLayout(note_box)
        self.note_edit = QTextEdit()
        note_layout.addWidget(self.note_edit)
        layout.addWidget(note_box)
        self.reload_scene_combo()
        self.on_scene_source_changed("taxonomy")
        return panel

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QWidget { background: #202225; color: #edf0f2; font-size: 14px; }
            QLineEdit, QTextEdit, QComboBox, QListWidget {
                background: #2c3035; border: 1px solid #474d55; border-radius: 6px;
                padding: 6px; selection-background-color: #2d7d9a;
            }
            QPushButton {
                background: #39515f; border: 1px solid #526b78; border-radius: 6px;
                padding: 7px 12px;
            }
            QPushButton:hover { background: #446575; }
            QGroupBox {
                border: 1px solid #444a52; border-radius: 8px; margin-top: 14px; padding: 10px;
                font-weight: 600;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
            QLabel#title { font-size: 24px; font-weight: 700; }
            QLabel#subtitle { color: #b9c0c7; }
            QLabel#hint { color: #aeb7bf; font-size: 12px; }
            QListWidget::item { padding: 6px; }
            QListWidget::item:selected { background: #2d7d9a; }
            """
        )

    def collect_import_data(self) -> dict[str, Any]:
        dataset_name = self.dataset_name_edit.text().strip()
        workspace = Path(self.workspace_edit.text().strip())
        taxonomy_path = Path(self.taxonomy_edit.text().strip())
        if not dataset_name or not self.workspace_edit.text().strip() or not taxonomy_path.is_file():
            raise ValueError("请检查数据集名称、工作区和场景体系路径。")

        import_type = self.import_type_combo.currentData()
        if import_type == "voc":
            if not self.voc_root_edit.text().strip():
                raise ValueError("请选择VOC根目录。")
            dataset_root = Path(self.voc_root_edit.text().strip())
            image_root = dataset_root
            mask_root = dataset_root
            mode = "voc_segmentation"
            image_dir_name = self.image_dir_name_edit.text().strip() or "JPEGImages"
            mask_dir_name = self.mask_dir_name_edit.text().strip() or "SegmentationClass"
            split = self.split_combo.currentText().strip() or "all"
            recursive = False
            image_scan_root = dataset_root / image_dir_name
            mask_scan_root = dataset_root / mask_dir_name
            if not dataset_root.is_dir() or not image_scan_root.is_dir() or not mask_scan_root.is_dir():
                raise ValueError("VOC根目录中没有找到图像目录或Mask目录。")
        else:
            if not self.image_root_edit.text().strip() or not self.mask_root_edit.text().strip():
                raise ValueError("请选择图像文件夹和Mask文件夹。")
            image_root = Path(self.image_root_edit.text().strip())
            mask_root = Path(self.mask_root_edit.text().strip())
            mode = str(self.generic_pairing_combo.currentData())
            image_dir_name = ""
            mask_dir_name = ""
            split = "all"
            recursive = self.recursive_checkbox.isChecked()
            image_scan_root = image_root
            mask_scan_root = mask_root
            if not image_root.is_dir() or not mask_root.is_dir():
                raise ValueError("请检查图像文件夹和Mask文件夹。")

        try:
            foreground_values = {
                int(value.strip())
                for value in self.mask_values_edit.text().split(",")
                if value.strip()
            }
        except ValueError as exc:
            raise ValueError("Mask前景标签值必须是整数，多个值使用英文逗号分隔。") from exc
        if not foreground_values:
            raise ValueError("请至少填写一个Mask前景标签值。")

        samples, warnings = scan_dataset(
            dataset_name,
            image_root,
            mask_root,
            mode,
            image_dir_name,
            mask_dir_name,
            split,
            recursive,
        )
        if not samples:
            raise ValueError("没有找到成功配对的image-mask样本。请检查路径和配对规则。")

        config = {
            "dataset_name": dataset_name,
            "import_type": import_type,
            "image_root": normalized_path(image_root),
            "mask_root": normalized_path(mask_root),
            "taxonomy_path": normalized_path(taxonomy_path),
            "pairing": {
                "mode": mode,
                "image_dir_name": image_dir_name,
                "mask_dir_name": mask_dir_name,
                "split": split,
                "recursive": recursive,
            },
            "mask": {
                "type": "binary",
                "background_value": 0,
                "foreground_values": sorted(foreground_values),
                "foreground_name": self.mask_name_edit.text().strip() or "foreground",
            },
        }
        return {
            "config": config,
            "samples": samples,
            "warnings": warnings,
            "inspection": inspect_sample_pairs(samples),
            "image_count": len(supported_files(image_scan_root, IMAGE_EXTS, recursive)),
            "mask_count": len(supported_files(mask_scan_root, MASK_EXTS, recursive)),
            "image_root": image_root,
            "mask_root": mask_root,
            "workspace": workspace,
            "taxonomy_path": taxonomy_path,
            "foreground_values": foreground_values,
        }

    def format_import_summary(self, data: dict[str, Any]) -> str:
        inspection = data["inspection"]
        values = inspection["mask_values"]
        value_text = ", ".join(str(value) for value in values) if values else "未能枚举"
        lines = [
            f"目录图像：{data['image_count']}    目录Mask：{data['mask_count']}",
            f"成功配对：{len(data['samples'])}    警告：{len(data['warnings'])}",
            f"抽样检查：{inspection['checked']} 对    Mask值：{value_text}",
            f"抽样尺寸不一致：{inspection['size_mismatches']}    读取错误：{inspection['read_errors']}",
        ]
        configured_values = data["foreground_values"]
        if values and not (configured_values & set(values)):
            lines.append("注意：配置的前景标签值没有出现在抽样Mask中。")
        if data["warnings"]:
            lines.append("部分警告：")
            lines.extend(f"- {warning}" for warning in data["warnings"][:4])
            if len(data["warnings"]) > 4:
                lines.append(f"- 其余 {len(data['warnings']) - 4} 条未显示")
        return "\n".join(lines)

    def show_import_preview_image(self, data: dict[str, Any]) -> None:
        sample = data["samples"][0]
        image = compose_preview(
            sample.image_path,
            sample.mask_path,
            True,
            45,
            data["foreground_values"],
        )
        pixmap = pil_to_qpixmap(image).scaled(
            self.preview_image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_image_label.setPixmap(pixmap)
        self.preview_image_label.setToolTip(
            f"image: {sample.image_path}\nmask: {sample.mask_path}"
        )

    def preview_import(self) -> None:
        try:
            data = self.collect_import_data()
            self.project_message.setText(self.format_import_summary(data))
            self.show_import_preview_image(data)
        except Exception as exc:
            self.project_message.setText(f"扫描失败：{exc}")
            self.preview_image_label.setPixmap(QPixmap())
            self.preview_image_label.setText("无法生成预览")
            QMessageBox.warning(self, "扫描失败", str(exc))

    def create_or_open_project(self) -> None:
        try:
            data = self.collect_import_data()
            config = data["config"]
            samples = data["samples"]
            warnings = data["warnings"]
            image_root = data["image_root"]
            mask_root = data["mask_root"]
            workspace = data["workspace"]
            taxonomy_path = data["taxonomy_path"]
            foreground_values = data["foreground_values"]
            dataset_name = config["dataset_name"]

            if (workspace / "review.sqlite3").exists():
                raise FileExistsError(
                    "该工作区已经存在审查数据库。继续以前的工作请使用“打开已有工作区”；"
                    "导入新数据集请更换工作区。"
                )
            summary = self.format_import_summary(data)
            answer = QMessageBox.question(
                self,
                "确认导入",
                f"{summary}\n\n确认建立审查项目吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

            self.taxonomy_path = taxonomy_path
            self.taxonomy = Taxonomy(taxonomy_path)
            self.mask_foreground_values = foreground_values
            workspace.mkdir(parents=True, exist_ok=True)
            (workspace / "exports").mkdir(exist_ok=True)
            (workspace / "cache").mkdir(exist_ok=True)
            (workspace / "project.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
            self.db = ReviewDatabase(workspace / "review.sqlite3")
            self.dataset_id = self.db.create_dataset(dataset_name, image_root, mask_root, config, self.taxonomy)
            self.db.add_samples(samples, self.dataset_id)
            value_text = ",".join(str(value) for value in sorted(foreground_values))
            foreground_name = config["mask"]["foreground_name"]
            msg = (
                f"导入完成：成功配对 {len(samples)} 个样本。"
                f"Mask 标签：0=背景，{value_text}={foreground_name}。"
            )
            if warnings:
                msg += f" 未配对或警告 {len(warnings)} 条，已跳过。"
            self.project_message.setText(msg)
            self.load_review_page()
        except Exception as exc:
            QMessageBox.critical(self, "导入失败", str(exc))

    def open_existing_workspace(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择已有工作区", self.workspace_edit.text())
        if not path:
            return
        db_path = Path(path) / "review.sqlite3"
        if not db_path.exists():
            QMessageBox.warning(self, "无法打开", "该工作区没有 review.sqlite3。")
            return
        self.db = ReviewDatabase(db_path)
        self.dataset_id = self.db.latest_dataset_id()
        if self.dataset_id is None:
            QMessageBox.warning(self, "无法打开", "数据库中没有数据集。")
            return
        row = self.db.latest_dataset()
        if row and row["config_json"]:
            config = json.loads(row["config_json"])
            taxonomy_path = config.get("taxonomy_path") or self.taxonomy_edit.text()
            if Path(taxonomy_path).exists():
                self.taxonomy = Taxonomy(Path(taxonomy_path))
            mask_config = config.get("mask", {})
            values = mask_config.get("foreground_values")
            self.mask_foreground_values = {int(value) for value in values} if values else None
        self.load_review_page()

    def load_review_page(self) -> None:
        if not self.db or self.dataset_id is None:
            return
        row = self.db.latest_dataset()
        self.dataset_label.setText(f"数据集：{row['name'] if row else self.dataset_id}")
        self.reload_samples()
        self.stack.setCurrentWidget(self.review_page)

    def reload_samples(self) -> None:
        if not self.db or self.dataset_id is None:
            return
        self.items = self.db.samples(self.dataset_id, self.filter_combo.currentText(), self.search_edit.text())
        self.sample_list.clear()
        for sample, review in self.items:
            scene = review.primary_level2_scene or review.custom_scene_name_en or "_"
            mark = {"accepted": "✓", "needs_correction": "!", "rejected": "x", "unreviewed": "·"}.get(review.quality_status, "·")
            item = QListWidgetItem(f"{mark} {Path(sample.image_path).name}\n{scene} | {review.scene_status}")
            item.setData(Qt.ItemDataRole.UserRole, sample.id)
            self.sample_list.addItem(item)
        self.update_common_scenes()
        self.status_label.setText(self.format_stats())
        if self.items:
            self.sample_list.setCurrentRow(0)
        else:
            self.image_canvas.set_error("当前过滤条件下没有样本")

    def on_sample_selected(self, row: int) -> None:
        if row < 0 or row >= len(self.items):
            return
        self.current_index = row
        sample, review = self.items[row]
        self.loading = True
        for button in self.quality_group.buttons():
            button.setChecked(button.text() == review.quality_status)
        source = review.scene_source if review.scene_source in SCENE_SOURCES else "taxonomy"
        self.scene_source_combo.setCurrentText(source)
        if review.level1_id:
            idx = self.domain_combo.findData(review.level1_id)
            if idx >= 0:
                self.domain_combo.setCurrentIndex(idx)
        self.reload_scene_combo()
        if review.primary_level2_scene:
            idx = self.scene_combo.findText(review.primary_level2_scene)
            if idx >= 0:
                self.scene_combo.setCurrentIndex(idx)
        self.custom_en_edit.setText(review.custom_scene_name_en)
        self.custom_zh_edit.setText(review.custom_scene_name_zh)
        self.mapping_edit.setText(review.taxonomy_mapping_suggestion)
        self.scene_status_combo.setCurrentText(review.scene_status)
        self.note_edit.setPlainText(review.reviewer_note)
        self.loading = False
        self.refresh_image()

    def refresh_image(self) -> None:
        if self.current_index < 0 or self.current_index >= len(self.items):
            return
        sample, _review = self.items[self.current_index]
        try:
            image = compose_preview(
                sample.image_path,
                sample.mask_path,
                self.mask_checkbox.isChecked(),
                self.opacity_slider.value(),
                self.mask_foreground_values,
            )
            self.image_canvas.set_pixmap(pil_to_qpixmap(image))
            self.path_label.setText(f"image: {sample.image_path}\nmask: {sample.mask_path}")
        except Exception as exc:
            self.image_canvas.set_error(str(exc))
            self.path_label.setText(f"读取失败：{sample.image_path}\n{exc}")

    def reload_scene_combo(self) -> None:
        if not hasattr(self, "scene_combo"):
            return
        domain_id = self.domain_combo.currentData()
        self.scene_combo.clear()
        self.scene_combo.addItem("")
        for scene in self.taxonomy.by_domain.get(domain_id, []):
            self.scene_combo.addItem(scene.scene)

    def on_scene_source_changed(self, source: str) -> None:
        is_taxonomy = source == "taxonomy"
        self.domain_combo.setEnabled(is_taxonomy)
        self.scene_combo.setEnabled(is_taxonomy)
        self.custom_en_edit.setEnabled(not is_taxonomy)
        self.custom_zh_edit.setEnabled(not is_taxonomy)
        self.mapping_edit.setEnabled(not is_taxonomy)

    def checked_quality_status(self) -> str:
        quality = "unreviewed"
        checked = self.quality_group.checkedButton()
        if checked:
            quality = checked.text()
        return quality

    def current_review_from_form(self) -> Review:
        quality = self.checked_quality_status()
        source = self.scene_source_combo.currentText()
        domain_id = self.domain_combo.currentData() or ""
        domain_text = self.domain_combo.currentText()
        level1_name = domain_text.split(" ", 1)[1].split(" / ")[0] if " " in domain_text else ""
        primary_scene = self.scene_combo.currentText() if source == "taxonomy" else ""
        custom_en = self.custom_en_edit.text().strip() if source != "taxonomy" else ""
        custom_zh = self.custom_zh_edit.text().strip() if source != "taxonomy" else ""
        scene_status = self.scene_status_combo.currentText()
        return Review(
            quality_status=quality,
            scene_status=scene_status,
            level1_id=domain_id if source == "taxonomy" else "",
            level1_name=level1_name if source == "taxonomy" else "",
            primary_level2_scene=primary_scene,
            scene_source=source,
            custom_scene_name_en=custom_en,
            custom_scene_name_zh=custom_zh,
            taxonomy_mapping_suggestion=self.mapping_edit.text().strip() if source != "taxonomy" else "",
            reviewer_note=self.note_edit.toPlainText().strip(),
        )

    def save_quality_status_only(self) -> None:
        if self.loading or not self.db or self.dataset_id is None:
            return
        if self.current_index < 0 or self.current_index >= len(self.items):
            return
        sample, old = self.items[self.current_index]
        review = Review(
            quality_status=self.checked_quality_status(),
            scene_status=old.scene_status,
            level1_id=old.level1_id,
            level1_name=old.level1_name,
            primary_level2_scene=old.primary_level2_scene,
            secondary_scenes_json=old.secondary_scenes_json,
            scene_source=old.scene_source,
            custom_scene_name_en=old.custom_scene_name_en,
            custom_scene_name_zh=old.custom_scene_name_zh,
            taxonomy_mapping_suggestion=old.taxonomy_mapping_suggestion,
            issue_tags_json=old.issue_tags_json,
            reviewer_note=old.reviewer_note,
        )
        self.db.save_review(sample.id, self.dataset_id, review)
        self.items[self.current_index] = (sample, review)
        self.update_sample_list_item(self.current_index, sample, review)
        self.status_label.setText(self.format_stats())

    def confirm_scene_selection(self) -> None:
        if self.current_index < 0 or self.current_index >= len(self.items):
            return
        source = self.scene_source_combo.currentText()
        if source == "taxonomy" and not self.scene_combo.currentText():
            QMessageBox.warning(self, "场景未选择", "请先选择一个正式二级场景。")
            return
        if source != "taxonomy" and not self.custom_en_edit.text().strip():
            QMessageBox.warning(self, "自定义场景未填写", "请填写自定义场景英文名。")
            return
        self.scene_status_combo.setCurrentText("assigned")
        self.save_current_review()
        self.scene_hint_label.setText("场景已确认并保存。")

    def reset_to_taxonomy_scene(self) -> None:
        self.scene_source_combo.setCurrentText("taxonomy")
        self.scene_combo.setCurrentIndex(0)
        self.custom_en_edit.clear()
        self.custom_zh_edit.clear()
        self.mapping_edit.clear()
        self.scene_status_combo.setCurrentText("unassigned")
        self.scene_hint_label.setText("已切换为正式体系选择，选好后点击确认场景。")

    def clear_scene_selection(self) -> None:
        self.scene_source_combo.setCurrentText("taxonomy")
        self.scene_combo.setCurrentIndex(0)
        self.custom_en_edit.clear()
        self.custom_zh_edit.clear()
        self.mapping_edit.clear()
        self.scene_status_combo.setCurrentText("unassigned")
        self.save_current_review()
        self.scene_hint_label.setText("当前样本场景已清空。")

    def save_current_review(self) -> None:
        if self.loading or not self.db or self.dataset_id is None:
            return
        if self.current_index < 0 or self.current_index >= len(self.items):
            return
        sample, _old = self.items[self.current_index]
        review = self.current_review_from_form()
        self.db.save_review(sample.id, self.dataset_id, review)
        self.items[self.current_index] = (sample, review)
        self.update_sample_list_item(self.current_index, sample, review)
        self.update_common_scenes()
        self.status_label.setText(self.format_stats())

    def update_sample_list_item(self, index: int, sample: Sample, review: Review) -> None:
        current = self.sample_list.item(index)
        if current:
            scene = review.primary_level2_scene or review.custom_scene_name_en or "_"
            mark = {"accepted": "✓", "needs_correction": "!", "rejected": "x", "unreviewed": "·"}.get(review.quality_status, "·")
            current.setText(f"{mark} {Path(sample.image_path).name}\n{scene} | {review.scene_status}")

    def update_common_scenes(self) -> None:
        if not self.db or self.dataset_id is None or not hasattr(self, "common_scene_list"):
            return
        self.common_scene_list.clear()
        for row in self.db.common_scenes(self.dataset_id):
            zh = f" / {row['scene_name_zh']}" if row["scene_name_zh"] else ""
            item = QListWidgetItem(f"{row['scene_name_en']}{zh}\n{row['scene_source']} | 已分配 {row['assigned_count']} 张")
            item.setData(Qt.ItemDataRole.UserRole, dict(row))
            self.common_scene_list.addItem(item)

    def apply_common_scene(self, item: QListWidgetItem) -> None:
        data = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(data, dict):
            return
        self.scene_source_combo.setCurrentText(data.get("scene_source", "taxonomy"))
        if data.get("scene_source") == "taxonomy":
            idx = self.domain_combo.findData(data.get("level1_id", ""))
            if idx >= 0:
                self.domain_combo.setCurrentIndex(idx)
            self.reload_scene_combo()
            idx2 = self.scene_combo.findText(data.get("scene_name_en", ""))
            if idx2 >= 0:
                self.scene_combo.setCurrentIndex(idx2)
        else:
            self.custom_en_edit.setText(data.get("scene_name_en", ""))
            self.custom_zh_edit.setText(data.get("scene_name_zh", ""))
            self.mapping_edit.setText(data.get("mapped_level2_scene", ""))
        self.scene_status_combo.setCurrentText("assigned")
        self.scene_hint_label.setText("已填入常用场景候选，可继续修改，点击确认场景后保存。")

    def prev_sample(self) -> None:
        self.save_current_review()
        if self.current_index > 0:
            self.sample_list.setCurrentRow(self.current_index - 1)

    def next_sample(self) -> None:
        self.save_current_review()
        if self.current_index + 1 < len(self.items):
            self.sample_list.setCurrentRow(self.current_index + 1)

    def next_unreviewed(self) -> None:
        self.save_current_review()
        for idx in range(self.current_index + 1, len(self.items)):
            if self.items[idx][1].quality_status == "unreviewed" or self.items[idx][1].scene_status == "unassigned":
                self.sample_list.setCurrentRow(idx)
                return
        QMessageBox.information(self, "完成", "当前列表中没有后续未审或未分场景样本。")

    def format_stats(self) -> str:
        if not self.db or self.dataset_id is None:
            return ""
        stats = self.db.stats(self.dataset_id)
        quality = stats["quality"]
        reviewed = stats["total"] - quality.get("unreviewed", 0)
        return f"已审 {reviewed} / {stats['total']} | accepted {quality.get('accepted', 0)} | rejected {quality.get('rejected', 0)}"

    def export_stats(self) -> None:
        if not self.db or self.dataset_id is None:
            return
        out_dir = QFileDialog.getExistingDirectory(self, "选择统计导出目录", "D:/硕士毕业论文/scene_review_tool")
        if not out_dir:
            return
        out = Path(out_dir)
        stats = self.db.stats(self.dataset_id)
        summary_path = out / "review_summary.csv"
        scene_path = out / "scene_statistics.csv"
        with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["item", "count"])
            writer.writerow(["total", stats["total"]])
            for key, value in stats["quality"].items():
                writer.writerow([key, value])
        with scene_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["scene", "source", "count"])
            writer.writerows(stats["scenes"])
        QMessageBox.information(self, "导出完成", f"已导出：\n{summary_path}\n{scene_path}")

    def export_by_scene(self) -> None:
        if not self.db or self.dataset_id is None:
            return
        out_dir = QFileDialog.getExistingDirectory(self, "选择场景导出目录", "D:/硕士毕业论文/scene_review_tool")
        if not out_dir:
            return
        root = Path(out_dir) / f"export_{time.strftime('%Y%m%d_%H%M%S')}"
        root.mkdir(parents=True, exist_ok=True)
        exported = 0
        rows: list[list[str]] = []
        for sample, review in self.db.samples(self.dataset_id, "all", ""):
            if review.quality_status != "accepted":
                continue
            scene = review.primary_level2_scene or review.custom_scene_name_en
            if not scene:
                continue
            safe_scene = "".join(c if c.isalnum() or c in " ._-" else "_" for c in scene).strip() or "_unassigned"
            scene_dir = root / safe_scene
            image_dir = scene_dir / "images"
            mask_dir = scene_dir / "masks"
            image_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)
            prefix = sample.id
            image_dst = image_dir / f"{prefix}_{Path(sample.image_path).name}"
            mask_dst = mask_dir / f"{prefix}_{Path(sample.mask_path).name}"
            shutil.copy2(sample.image_path, image_dst)
            shutil.copy2(sample.mask_path, mask_dst)
            rows.append([sample.id, scene, review.scene_source, sample.image_path, sample.mask_path, str(image_dst), str(mask_dst)])
            exported += 1
        with (root / "export_manifest.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sample_id", "scene", "scene_source", "source_image", "source_mask", "export_image", "export_mask"])
            writer.writerows(rows)
        QMessageBox.information(self, "导出完成", f"已导出 {exported} 个 accepted 且已分场景样本到：\n{root}")

    def keyPressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        key = event.key()
        if key in (Qt.Key.Key_Right, Qt.Key.Key_D):
            self.next_sample()
        elif key in (Qt.Key.Key_Left, Qt.Key.Key_A):
            self.prev_sample()
        elif key == Qt.Key.Key_Space:
            self.mask_checkbox.setChecked(not self.mask_checkbox.isChecked())
        elif key == Qt.Key.Key_F:
            self.image_canvas.fit()
        elif key == Qt.Key.Key_1:
            self.set_quality("accepted")
        elif key == Qt.Key.Key_2:
            self.set_quality("needs_correction")
        elif key == Qt.Key.Key_3:
            self.set_quality("rejected")
        else:
            super().keyPressEvent(event)

    def set_quality(self, status: str) -> None:
        for button in self.quality_group.buttons():
            if button.text() == status:
                button.setChecked(True)
                self.save_current_review()
                return


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
