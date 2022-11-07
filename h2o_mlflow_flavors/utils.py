import os
import zipfile
import re

from shutil import make_archive

def unzip_specific_file(zip_file_name, *extract_file_name, directory="/tmp"):
    with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
        for file_name in extract_file_name:
                zip_ref.extract(file_name, directory)


def unzip_specific_folder(zip_file_name, *extract_folder_name, directory="/tmp"):
    with zipfile.ZipFile(zip_file_name) as archive:
        for folder_name in extract_folder_name:
            names_foo = [i for i in archive.namelist() if i.startswith(folder_name)]
            for file in names_foo:
                archive.extract(file, directory)


def match_file_from_name_pattern(zip_file_name, pattern):
    with zipfile.ZipFile(zip_file_name) as archive:
        for info in archive.infolist():
            if re.match(pattern, info.filename):
                return info.filename
    return None

def zip_folder(source_folder_path, output_zip_filename):
    return make_archive(output_zip_filename, "zip", source_folder_path)
