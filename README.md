# Lean-SLD-TabICL
Lean-SLD-TabICL code
# TabICL-CDSS for Lean SLD

This repository contains the clinical decision support system (CDSS) source code and the trained TabICL model for our paper: 
**"Interpretable deep learning with TabICL foundation model enhances lean steatotic liver disease screening: a cross-national validation study"**.

## Files Included:
* `index7.html`: The interactive offline Graphical User Interface (GUI).
* `main7.py`: The core Python script for data processing and AI inference.
* `Champion_TabICL.pkl`: The trained TabICL deep learning model weights.
* `Features.pkl` & `Imputer.pkl`: Configuration files for feature processing and data imputation.

## How to Run the CDSS:
1. Download all files in this repository to a single local folder.
2. Ensure you have Python installed, along with the necessary data science packages (e.g., `pandas`, `scikit-learn`, `xgboost`, `shap`). 
3. Run the `main7.py` script to start the local backend.
4. Double-click the `index7.html` file to open the calculator in your local web browser.
5. Input the 8 routine clinical parameters to get real-time risk stratification.

## Contact
For any questions regarding the dataset or the code, please contact: yk1998@stu.xjtu.edu.cn
