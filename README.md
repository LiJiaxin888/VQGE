# **Project Description**

This repository contains the implementation code for the paper "Variational quantum algorithm for generalized eigenvalue problems of non-Hermitian systems". We are providing this code to ensure the reproducibility of our results during the review process. Please be aware that this code is still under active development and refinement. 

This project is developed based on **pybind11 + Python 3.9**, and provides precompiled `.whl` files for basic functionalities, which can be installed and used in combination with secondary encapsulated packages.

------

## Install Python and Python packages

1. Download and install [Anaconda](https://www.anaconda.com/download)

2. Open the Anaconda Prompt

3. Create a virtual environment and activate it with Python 3.9.11 as an example

   ```
   conda create -n quantum python=3.9.11 -y
   conda activate quantum
   ```

4. Install Python packages

   ```
   pip install pyqpanda3
   ```

------

## Environment Dependencies

```python
from pyqpanda3.core import *
import random
import numpy as np
import math
import scipy.linalg as la
from scipy.linalg import eig
```

---

## **Installation Process**

### 1. **Install the base `.whl` file**

**Install the provided `.whl` file as the base library**：

```bash
pip install pyvqges-1.0.0-py3-none-any.whl
```

### 2. **Install the secondary encapsulated package**

Navigate to the package folder and execute：

```bash
cd package
pip install .
```

---

## **Usage Example**

```python
import numpy as np
from vqgespy import *

A = np.random.rand(32, 32)
B = np.random.rand(32, 32)
result = VQGES(A, B)
print(result)
```

---

## **Visualization**

After running the main program, we can generate visualization plots of the iteration process using the `draw.py` script

------

## **Notes**

* It is recommended to use within a virtual environment (e.g., `venv` or `conda`).
* If multiple Python versions are installed, ensure that `pip` corresponds to Python 3.9.
* If encountering dependency issues while installing the `.whl` file, please update `pip` first.

---

## **Contact**

For questions or feedback, please contact the project developers.

