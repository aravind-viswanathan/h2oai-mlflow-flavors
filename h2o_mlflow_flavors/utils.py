import os
import zipfile
import re

from shutil import make_archive

def unzip_specific_file(zip_file_name, extract_file_name, directory="/tmp"):
    with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
        zip_ref.extract(extract_file_name, directory)

    return directory+"/"+extract_file_name

def unzip_specific_folder(zip_file_name, extract_folder_name, directory="/tmp"):
    with zipfile.ZipFile(zip_file_name) as archive:
        names_foo = [i for i in archive.namelist() if i.startswith(extract_folder_name)]
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
