from coder_review_benchmark.data import multi_swe_image_name


def test_multi_swe_image_name_matches_official_convention():
    assert multi_swe_image_name({"org": "BurntSushi", "repo": "ripgrep", "number": 2209}) == (
        "mswebench/burntsushi_m_ripgrep:pr-2209"
    )
