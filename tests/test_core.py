from pathlib import Path

from PIL import Image

from scene_review_tool.app import Review, ReviewDatabase, Sample, Taxonomy, compose_preview, scan_dataset


def test_taxonomy_loads_project_taxonomy():
    taxonomy = Taxonomy(Path("D:/硕士毕业论文/ppt相关/geo_scene_taxonomy_1.0.json"))
    assert taxonomy.version == "1.0"
    assert taxonomy.find_scene("river") is not None


def test_mirrored_subdirectories_scan(tmp_path):
    dataset = tmp_path / "dataset"
    image_dir = dataset / "group1" / "images"
    mask_dir = dataset / "group1" / "labels"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8), (10, 20, 30)).save(image_dir / "a.png")
    Image.new("L", (8, 8), 255).save(mask_dir / "a.png")

    samples, warnings = scan_dataset("demo", dataset, dataset, "mirrored_subdirectories", "images", "labels")

    assert len(samples) == 1
    assert warnings == []
    assert samples[0].parent_group_id == "group1"


def test_voc_segmentation_scan_uses_split_file(tmp_path):
    dataset = tmp_path / "voc"
    image_dir = dataset / "JPEGImages"
    mask_dir = dataset / "SegmentationClass"
    split_dir = dataset / "ImageSets" / "Segmentation"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    split_dir.mkdir(parents=True)
    for name in ("a", "b"):
        Image.new("RGB", (8, 8), (10, 20, 30)).save(image_dir / f"{name}.jpg")
        Image.new("L", (8, 8), 1).save(mask_dir / f"{name}.png")
    (split_dir / "test.txt").write_text("b\n", encoding="utf-8")

    samples, warnings = scan_dataset(
        "voc-demo", dataset, dataset, "voc_segmentation", "JPEGImages", "SegmentationClass", "test"
    )

    assert warnings == []
    assert len(samples) == 1
    assert Path(samples[0].image_path).stem == "b"
    assert Path(samples[0].mask_path).stem == "b"
    assert samples[0].parent_group_id == "test"


def test_generic_relative_path_scan_pairs_duplicate_stems_safely(tmp_path):
    image_root = tmp_path / "images"
    mask_root = tmp_path / "masks"
    for group in ("north", "south"):
        (image_root / group).mkdir(parents=True)
        (mask_root / group).mkdir(parents=True)
        Image.new("RGB", (8, 8), (10, 20, 30)).save(image_root / group / "tile.jpg")
        Image.new("L", (8, 8), 1).save(mask_root / group / "tile.png")

    samples, warnings = scan_dataset(
        "relative-demo", image_root, mask_root, "relative_path_stem", "", "", "all", True
    )

    assert len(samples) == 2
    assert warnings == []
    assert {Path(sample.image_path).parent.name for sample in samples} == {"north", "south"}


def test_generic_same_stem_does_not_guess_ambiguous_pairs(tmp_path):
    image_root = tmp_path / "images"
    mask_root = tmp_path / "masks"
    for group in ("north", "south"):
        (image_root / group).mkdir(parents=True)
        (mask_root / group).mkdir(parents=True)
        Image.new("RGB", (8, 8), (10, 20, 30)).save(image_root / group / "tile.jpg")
        Image.new("L", (8, 8), 1).save(mask_root / group / "tile.png")

    samples, warnings = scan_dataset(
        "ambiguous-demo", image_root, mask_root, "same_stem", "", "", "all", True
    )

    assert samples == []
    assert warnings == ["ambiguous duplicate pairing key: tile"]


def test_binary_mask_value_one_is_rendered_as_foreground(tmp_path):
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    Image.new("RGB", (2, 1), (10, 20, 30)).save(image_path)
    mask = Image.new("L", (2, 1), 0)
    mask.putpixel((1, 0), 1)
    mask.save(mask_path)

    preview = compose_preview(str(image_path), str(mask_path), True, 50, {1}).convert("RGB")

    assert preview.getpixel((0, 0)) == (10, 20, 30)
    assert preview.getpixel((1, 0)) != (10, 20, 30)


def test_database_persists_samples(tmp_path):
    db = ReviewDatabase(tmp_path / "review.sqlite3")
    taxonomy = Taxonomy(Path("D:/硕士毕业论文/ppt相关/geo_scene_taxonomy_1.0.json"))
    dataset_id = db.create_dataset("demo", tmp_path, tmp_path, {"pairing": {}}, taxonomy)
    assert dataset_id == 1
    db.close()


def test_common_scene_count_is_assigned_sample_count(tmp_path):
    db = ReviewDatabase(tmp_path / "review.sqlite3")
    taxonomy = Taxonomy(Path("D:/硕士毕业论文/ppt相关/geo_scene_taxonomy_1.0.json"))
    dataset_id = db.create_dataset("demo", tmp_path, tmp_path, {"pairing": {}}, taxonomy)
    sample = Sample("s1", "image.png", "mask.png", "group")
    db.add_samples([sample], dataset_id)
    review = Review(
        quality_status="accepted",
        scene_status="assigned",
        level1_id="G1",
        level1_name="水系与湿地",
        primary_level2_scene="river",
        scene_source="taxonomy",
    )

    db.save_review(sample.id, dataset_id, review)
    db.save_review(sample.id, dataset_id, review)

    common = db.common_scenes(dataset_id)
    assert len(common) == 1
    assert common[0]["scene_name_en"] == "river"
    assert common[0]["assigned_count"] == 1
    db.close()
