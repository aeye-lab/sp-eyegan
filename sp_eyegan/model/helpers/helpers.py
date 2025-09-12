import os


def write_json_file(contents, destination_filepath):
    """
    Writes dictionary Contents into json file
    :param contents: source dictionary
    :param destination_filepath: file path to write json file
    :return: 1
    """
    import json
    with open(destination_filepath, 'w') as fp:
        json.dump(contents, fp)
    return 1


def load_json_file(source_filepath):
    """
    Loads and returns content of a json file
    :param source_filepath: source file path
    :return: contents of json file
    """
    import json
    F = open(source_filepath)
    file = F.read()
    F.close()
    content = json.loads(file)
    return content
