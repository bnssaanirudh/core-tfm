from core_tfm.data.openml_specs import CLASSIFICATION_DATASETS


def test_final_benchmark_specs_use_adult_not_infeasible_marketing():
    assert "adult" in CLASSIFICATION_DATASETS
    assert CLASSIFICATION_DATASETS["adult"] == {
        "openml_id": 1590,
        "target_a": "class",
        "target_b": "marital-status",
    }
    assert "marketing" not in CLASSIFICATION_DATASETS
