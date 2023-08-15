import json


def get_settings(file_name: str):
    with open(file_name, 'r') as file:
        settings = json.load(file)
    return settings
