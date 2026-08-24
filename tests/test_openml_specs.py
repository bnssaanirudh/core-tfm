from core_tfm.data.openml_specs import CLASSIFICATION_DATASETS


def test_completed_bounded_benchmark_specs_use_marketing():
    assert "marketing" in CLASSIFICATION_DATASETS
    assert CLASSIFICATION_DATASETS["marketing"] == {
        "openml_id": 46940,
        "target_a": "Response",
        "target_b": "Marital_Status",
    }
    assert "adult" not in CLASSIFICATION_DATASETS
