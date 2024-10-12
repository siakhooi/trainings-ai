import zipfile
import os

parent_path = "level-up-python-3210418-main/src/14 Build a Zip Archive"


def zip_all(directory, file_extensions, zip_filename):

    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            _, extension = os.path.splitext(filename)
            if extension in file_extensions:
                files.append(os.path.join(root, filename))

    with zipfile.ZipFile(zip_filename, "w") as zip_file:
        for file in files:
            zip_file.write(file, arcname=os.path.relpath(file, directory))

# commands used in solution video for reference
if __name__ == "__main__":
    zip_all(f"./{parent_path}/my_stuff", [".jpg", ".txt"], "my_stuff.zip")
