"""Historical results retain their numerical provenance under the new label."""

import pytest

from src.evaluation.registry import normalize_report_labels


@pytest.mark.parametrize("old_name", ["ICR-MMSE-Net", "ICR-MMSE", "CRU-Net"])
def test_historical_report_label_is_normalized_without_mutation(old_name):
    curve = {"identifier": "icr_mmse", "ber": [0.1], "provenance": {"code": "historical"}}
    report = {"curves": {old_name: curve}, "comparison_algorithms": [old_name]}
    renamed = normalize_report_labels(report)
    assert renamed["curves"]["CRENet"]["identifier"] == "crenet"
    assert renamed["curves"]["CRENet"]["provenance"] == curve["provenance"]
    assert renamed["comparison_algorithms"] == ["CRENet"]
    assert report["curves"][old_name]["identifier"] == "icr_mmse"


def test_historical_and_current_labels_cannot_hide_duplicate_curves():
    with pytest.raises(ValueError, match="Duplicate report curve"):
        normalize_report_labels({"curves": {"ICR-MMSE-Net": {}, "CRENet": {}}})
