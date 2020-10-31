## Deployment of tensorflow, keras, sklearn, statmodels on vs-code by miniconda

```
conda create --name TF
conda activate TF
conda install -c anaconda pandas xlrd xlwt seaborn scikit-learn pillow keras-gpu keras ipykernel pylint
```

## Remark
To write file to disk, we should get permission. It can be done by

```
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
```