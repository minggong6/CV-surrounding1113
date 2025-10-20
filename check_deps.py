"""
Smart dependency checker for this workspace.

This script attempts to import modules by their import names and reports the installed
version (when available). It also compares against expected versions inferred from
the pip list you provided.

Run with: python check_deps.py

The mapping below was populated from the pip list you pasted into the chat; packages
present there are set as the "expected" versions. Missing packages will show expected
version as None.
"""

import importlib
import pkg_resources
import sys

# Map importable module name -> (pypi package name, expected_version_or_None)
# Expected versions were taken from the pip list you provided in the previous message.
MODULES = {
    # core
    'numpy': ('numpy', '1.19.5'),
    'scipy': ('scipy', '1.5.3'),
    'skimage': ('scikit-image', '0.17.2'),
    'SimpleITK': ('SimpleITK', '2.1.1.2'),
    'nibabel': ('nibabel', '4.0.2'),
    # connected-components-3d is the distribution name for cc3d functionality
    'cc3d': ('connected-components-3d', '3.6.0'),
    'tqdm': ('tqdm', '4.64.1'),
    'xlrd2': ('xlrd2', '1.3.4'),
    'xlwt': ('xlwt', '1.3.0'),

    # imaging / io
    'cv2': ('opencv-python', '4.4.0'),
    'PIL': ('Pillow', '8.4.0'),
    'matplotlib': ('matplotlib', '3.3.4'),
    'imageio': ('imageio', '2.15.0'),

    # advanced / optional
    'kimimaro': ('kimimaro', '3.3.0'),
    'cloudvolume': ('cloud-volume', '8.8.2'),
    'itk': ('itk', '5.2.1.post1'),

    # optional commonly-used packages (not present in your pip output)
    'sklearn': ('scikit-learn', None),
}


def get_version_from_module(mod):
    # Try common version attributes
    for attr in ('__version__', 'VERSION', 'version'):
        v = getattr(mod, attr, None)
        if isinstance(v, str):
            return v
        if isinstance(v, (tuple, list)):
            return '.'.join(map(str, v))
    return None


def get_version_from_pkg(package_name):
    try:
        return pkg_resources.get_distribution(package_name).version
    except Exception:
        return None


def main():
    print('Checking modules and comparing to expected versions (if provided)')
    missing = []
    mismatched = []
    for import_name, (pkg_name, expected) in MODULES.items():
        try:
            mod = importlib.import_module(import_name)
            inst_ver = get_version_from_module(mod) or get_version_from_pkg(pkg_name)
            if inst_ver is None:
                status = f'OK: imported {import_name} (version unknown)'
            else:
                status = f'OK: imported {import_name} (installed: {inst_ver})'
            if expected:
                if inst_ver is None:
                    status += f'  | expected: {expected}'
                elif inst_ver != expected:
                    status += f'  | expected: {expected}  <-- version differs'
                    mismatched.append((import_name, pkg_name, expected, inst_ver))
                else:
                    status += f'  | expected: {expected}  (matches)'
            print(status)
        except Exception as e:
            print(f'MISSING: {import_name} (package: {pkg_name})  | expected: {expected}')
            missing.append((import_name, pkg_name, expected))

    if missing:
        print('\nSummary: missing packages/modules:')
        for imp, pkg, exp in missing:
            if exp:
                print(f'  - {imp} (package {pkg})  ; suggested install: {pkg}=={exp}')
            else:
                print(f'  - {imp} (package {pkg})  ; suggested install: {pkg} (no preferred version)')

    if mismatched:
        print('\nSummary: version mismatches:')
        for imp, pkg, exp, inst in mismatched:
            print(f'  - {imp} (package {pkg})  installed: {inst}  expected: {exp}')

    if not missing and not mismatched:
        print('\nAll checked modules are importable and match expected versions where provided.')


if __name__ == '__main__':
    main()
