#!/usr/bin/env python3
"""Inspect DB-backed annotations before building a YOLO dataset version."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from build_data_v2_from_db import (
    COPY_TABLES,
    annotation_has_allowed_label,
    annotation_is_selected,
    class_names_from_backup,
    format_count_map,
    is_nonempty_text,
    parse_int,
    read_classes_file,
    read_image_roots,
    resolve_image_path,
    table_rows,
    validate_yolo_text,
)
from db_copy_parser import load_tables


def format_classes(class_names: list[str]) -> str:
    """Format class names with numeric IDs."""
    if not class_names:
        return "None"
    return "\n".join(f"  {index}: {name}" for index, name in enumerate(class_names))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect labeling DB annotations and image-path resolution."
    )
    parser.add_argument("--backup-root", required=True, help="Backup root containing labeling_db.sql.gz.")
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--image-root-override", action="append", default=[])
    parser.add_argument("--image-roots-file")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--include-draft", action="store_true")
    parser.add_argument("--allow-empty-labels", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    backup_root = Path(args.backup_root).expanduser().resolve()
    db_path = backup_root / "labeling_db.sql.gz"

    if not db_path.exists():
        print(f"ERROR: DB dump does not exist: {db_path}", file=sys.stderr)
        return 1

    try:
        image_roots = read_image_roots(args.image_root_override, args.image_roots_file)
        tables = load_tables(db_path, COPY_TABLES)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    projects = table_rows(tables, "projects")
    frames = table_rows(tables, "frames")
    annotations = table_rows(tables, "annotations")
    project_row = next(
        (row for row in projects if parse_int(row.get("id"), -1) == args.project_id),
        None,
    )
    class_names, class_source, class_warnings = class_names_from_backup(
        backup_root,
        project_row,
        args.project_id,
    )

    print("Project summary")
    print(f"  project_id: {args.project_id}")
    print(f"  project_name: {project_row.get('name') if project_row else 'unknown'}")
    print(f"  project_status: {project_row.get('status') if project_row else 'unknown'}")
    print(f"  class_source: {class_source}")
    print("  classes:")
    print(format_classes(class_names))
    for warning in class_warnings:
        print(f"  class_warning: {warning}")

    classes_path = backup_root / "labels_working" / f"project_{args.project_id}" / "classes.txt"
    if classes_path.exists():
        try:
            working_classes = read_classes_file(classes_path)
        except OSError as exc:
            print(f"  labels_working classes: unreadable: {exc}")
        else:
            print(f"  labels_working classes path: {classes_path}")
            print(format_classes(working_classes))

    project_frames = [
        frame for frame in frames if parse_int(frame.get("project_id"), -1) == args.project_id
    ]
    frame_by_id = {str(frame.get("id")): frame for frame in project_frames if frame.get("id")}
    project_annotations = [
        annotation
        for annotation in annotations
        if annotation.get("frame_id") is not None and str(annotation.get("frame_id")) in frame_by_id
    ]

    frame_status_counts = Counter(str(frame.get("status") or "") for frame in project_frames)
    review_status_counts = Counter(
        str(annotation.get("review_status") or "") for annotation in project_annotations
    )
    submitted_non_null = sum(1 for annotation in project_annotations if annotation.get("submitted_at"))
    non_empty_yolo = sum(
        1 for annotation in project_annotations if is_nonempty_text(annotation.get("yolo_text"))
    )
    selected_annotations = [
        annotation
        for annotation in project_annotations
        if annotation_is_selected(annotation, args.include_draft)
        and annotation_has_allowed_label(annotation, args.allow_empty_labels)
    ]

    validation_errors: list[str] = []
    class_instance_counts: Counter[int] = Counter()
    if class_names:
        for annotation in selected_annotations:
            frame_id = str(annotation.get("frame_id") or "")
            annotation_id = str(annotation.get("id") or "")
            errors, counts = validate_yolo_text(
                annotation.get("yolo_text"),
                len(class_names),
                f"annotation {annotation_id} frame {frame_id}",
            )
            validation_errors.extend(errors)
            class_instance_counts.update(counts)

    print("\nCounts")
    print(f"  project_frames: {len(project_frames)}")
    print("  frame_status_counts:")
    print(format_count_map(frame_status_counts))
    print(f"  joined_annotations: {len(project_annotations)}")
    print("  annotation_review_status_counts:")
    print(format_count_map(review_status_counts))
    print(f"  submitted_at_non_null: {submitted_non_null}")
    print(f"  non_empty_yolo_text: {non_empty_yolo}")
    print(f"  selected_annotations: {len(selected_annotations)}")
    print(f"  yolo_validation_errors: {len(validation_errors)}")
    print("  class_instances:")
    if class_names:
        print(
            format_count_map(
                {
                    f"{class_id}: {class_names[class_id]}": class_instance_counts.get(class_id, 0)
                    for class_id in range(len(class_names))
                }
            )
        )
    else:
        print("No classes available; skipped class instance counting.")

    sample_size = max(args.sample_size, 0)
    sampled_annotations = selected_annotations[:sample_size]
    direct_resolved = 0
    override_resolved: Counter[str] = Counter()
    missing_examples: list[str] = []
    sampled_paths: list[str] = []
    joined_examples: list[str] = []

    for annotation in sampled_annotations:
        frame = frame_by_id[str(annotation.get("frame_id"))]
        image_path = str(frame.get("image_path") or "")
        sampled_paths.append(image_path)
        resolved_path, method, root = resolve_image_path(image_path, image_roots)
        if method == "direct":
            direct_resolved += 1
        elif method == "override" and root is not None:
            override_resolved[root] += 1
        else:
            missing_examples.append(
                f"frame_id={frame.get('id')} annotation_id={annotation.get('id')} image_path={image_path}"
            )

        if len(joined_examples) < 20:
            first_yolo_line = ""
            if annotation.get("yolo_text"):
                first_yolo_line = str(annotation["yolo_text"]).splitlines()[0]
            joined_examples.append(
                "annotation_id={annotation_id} frame_id={frame_id} video_id={video_id} "
                "frame_index={frame_index} image_path={image_path} "
                "resolved_image_path={resolved_image_path} first_yolo_line={first_yolo_line}".format(
                    annotation_id=annotation.get("id") or "",
                    frame_id=frame.get("id") or "",
                    video_id=frame.get("video_id") or "",
                    frame_index=frame.get("frame_index") or "",
                    image_path=image_path,
                    resolved_image_path=str(resolved_path) if resolved_path else "",
                    first_yolo_line=first_yolo_line,
                )
            )

    print("\nImage resolution sample")
    print(f"  requested_sample_size: {sample_size}")
    print(f"  sampled_rows: {len(sampled_annotations)}")
    print("  sampled_paths:")
    for sampled_path in sampled_paths[:20]:
        print(f"  - {sampled_path}")
    if len(sampled_paths) > 20:
        print(f"  - ... {len(sampled_paths) - 20} more not shown")
    print(f"  direct_resolved_count: {direct_resolved}")
    print("  resolved_count_per_override_root:")
    print(format_count_map(override_resolved))
    print(f"  missing_count: {len(missing_examples)}")
    print("  first_20_missing_examples:")
    for missing in missing_examples[:20]:
        print(f"  - {missing}")
    if not missing_examples:
        print("  None")

    print("\nSample joined rows")
    for example in joined_examples:
        print(f"  - {example}")
    if validation_errors:
        print("\nFirst 20 validation errors")
        for error in validation_errors[:20]:
            print(f"  - {error}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
