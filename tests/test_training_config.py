from training import config


def test_training_seasons_covers_the_intended_scope():
    assert config.TRAINING_SEASONS == ("2023-24", "2024-25", "2025-26")


def test_training_seasons_is_an_immutable_sequence():
    assert isinstance(config.TRAINING_SEASONS, tuple)


def test_raw_data_dir_lives_under_training_data_raw():
    parts = config.RAW_DATA_DIR.parts[-3:]
    assert parts == ("training", "data", "raw")
