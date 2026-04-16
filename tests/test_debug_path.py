import src.main_execution as me


def test_file_path():
    print(f"\nModule file: {me.__file__}")
    print(f"Attributes: {dir(me)}")
    assert hasattr(me, "fetch_mt5_training_data")
