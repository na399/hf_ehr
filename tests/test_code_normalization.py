from hf_ehr.tokenizers.code_normalization import clmbr_meds_code_candidates, clmbr_meds_raw_variants


def test_clmbr_meds_code_candidates_include_single_slash_vocab_code():
    assert clmbr_meds_code_candidates("LOINC//3023314") == ("LOINC//3023314", "LOINC/3023314")
    assert clmbr_meds_code_candidates("RxNorm//46287424//end") == (
        "RxNorm//46287424//end",
        "RxNorm/46287424",
    )


def test_clmbr_meds_raw_variants_include_interval_boundaries():
    assert clmbr_meds_raw_variants("SNOMED/4145513") == (
        "SNOMED/4145513",
        "SNOMED//4145513",
        "SNOMED//4145513//start",
        "SNOMED//4145513//end",
    )
