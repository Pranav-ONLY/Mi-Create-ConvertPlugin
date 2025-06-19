# Wheel installer
# Automatically installs .whl files placed in the libs folder
# Also keeps tracks of libs extracted from the .whl files

import os
import json
import shutil
import zipfile

from plugin_api import PluginAPI

# Initialize paths and API
plugin_dir = os.path.dirname(os.path.realpath(__file__))
libs_dir = os.path.join(plugin_dir, "libs")
api = PluginAPI()

def install():
    # Keep track of all modules in the wheel
    module_list = []

    if os.path.isfile(os.path.join(libs_dir, "libs.json")):
        # Wheels been installed, no need to do it again
        return
    
    if os.path.isdir(libs_dir) is not True:
        # No libs folder, there's nothing to install
        return

    for lib in os.listdir(libs_dir):
        # Initialize ZipFile object to get a list of all modules
        # wheel = zipfile.ZipFile(os.path.join(libs_dir, lib))
        # module_list = module_list + wheel.namelist()

        with zipfile.ZipFile(os.path.join(libs_dir, lib), 'r') as wheel:
            module_list = module_list + wheel.namelist()

        # Call PluginAPI install_library with the wheel
        api.install_library(os.path.join(libs_dir, lib))

        # Delete the wheel file
        os.remove(os.path.join(libs_dir, lib))

    # Write the module list to file
    with open(os.path.join(libs_dir, "libs.json"), 'w+') as f:
        json.dump(module_list, f)

def uninstall():
    if os.path.isdir(libs_dir) is not True:
        # No libs folder, there's nothing to uninstall
        return

    # Check if libs.json is a valid file
    if os.path.isfile(os.path.join(libs_dir, "libs.json")):
        # Open the libs.json file and convert to a Python object
        with open(os.path.join(libs_dir, "libs.json")) as libs:
            module_list = json.load(libs)

        # Delete all files that were installed
        dirs = list(set([os.path.dirname(x) for x in module_list]))
        for folder in dirs:
            if folder != "":
                if os.path.exists(folder):
                    api.delete_library(folder)

        os.remove(os.path.join(libs_dir, "libs.json"))
    else:
        # No wheels have been installed, or someone deleted the libs folder
        api.showDialog("warning", "Failed to get installed libraries list (libs.json), you may need to delete plugin libraries manually. This is not required.")