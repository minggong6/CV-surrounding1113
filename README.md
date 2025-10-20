Pancreas Resectability Demo

Quick Streamlit prototype that wraps existing pipeline functions in this repository.

Usage (local):

1. Create/activate a Python environment (recommend Conda on Windows):

```powershell
conda create -n pancreas python=3.10 -y
conda activate pancreas
# Install the heavy/compiled dependencies from conda-forge first
conda install -c conda-forge kimimaro cloud-volume itk connected-components-3d -y
# Then install the remaining (pure Python) dependencies
pip install -r requirements.txt
```

2. Run the Streamlit app:

```powershell
streamlit run app_streamlit.py
```

Notes:
- The repository's skeleton analysis depends on `kimimaro` and `cloud-volume`; these may be easier to install using conda-forge.
  Example (recommended full install):

```powershell
conda install -c conda-forge kimimaro cloud-volume itk connected-components-3d -y
```
- The app expects segmented NIfTI files where labels are: 1=artery, 2=tumor, 3=vein.
- This prototype uses a simple rule-based resectability score. Replace it with a trained model if you have labeled data.
