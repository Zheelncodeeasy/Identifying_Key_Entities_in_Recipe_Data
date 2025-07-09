# Identifying_Key_Entities_in_Recipe_Data


## 1. Objective

To build a **Conditional Random Field (CRF)** model for extracting structured entities — **quantity**, **unit**, and **ingredient** — from unstructured recipe ingredient data.


## 2. Summary of Process and Findings

### a. Data Loading and Preparation
- JSON data was loaded into a Pandas DataFrame.
- Rows with mismatched token-label lengths were identified and removed.
- Data was split into:
  - **Training set**: 70%  
  - **Validation set**: 30%


### b. Exploratory Data Analysis (EDA)
- Most frequent entities observed:
  - **Ingredients**: `"powder"`, `"Salt"`, `"seeds"`
  - **Units**: `"teaspoon"`, `"cup"`, `"tablespoon"`
- Label distribution showed **class imbalance**:
  - `ingredient` → most frequent  
  - `quantity` and `unit` → less frequent
- Visualizations supported these patterns in both training and validation sets.


### c. Feature Extraction
- Custom `word2features` function created using **spaCy**.
- Included:
  - Token-level features  
  - Contextual features  
  - Pattern-based indicators (for numbers, units, etc.)


### d. Class Weighting
- **Inverse frequency** method used for class weighting.
- Additional weight applied to underrepresented classes (`unit`, `quantity`) to reduce model bias.


### e. Model Building and Training
- CRF model configured with:
  - Algorithm: **L-BFGS**
  - Regularization: **L1 and L2**
  - Max iterations set
  - Enabled **all possible transitions**
- Model trained on **weighted training data**.


## Evaluation Results

### a. Training Set Performance
- **Accuracy**: 97%
- High **precision**, **recall**, and **f1-score** for all classes
- Indicates strong learning from the training data


### b. Validation Set Performance
- **Overall Accuracy**: 96.42%
- Per-class performance:
  - `ingredient`: **98.34%**
  - `quantity`: **94.89%**
  - `unit`: **86.87%** (lowest)


### c. Confusion Matrix Observations
- Minimal errors for `ingredient`
- Common misclassifications:
  - `unit` vs `ingredient`
  - `quantity` vs `ingredient`


### d. Error Analysis
- Frequent issues:
  - **Ambiguous tokens** like `"cloves"`, `"teaspoon"`
  - **Punctuation** symbols (e.g., `")"`) within ingredient phrases
  - **Numeric tokens** like `"1/2"` misclassified due to weak context
- Indicates that local context alone may be insufficient for perfect classification


## 4. Conclusion

- CRF model is effective for recipe entity recognition
- **Performs well** on `ingredient` and `quantity` classes
- **Struggles more** with the `unit` class due to ambiguity and low frequency


## Final Verdict

The CRF pipeline — including data cleaning, feature engineering, and class weighting — builds a **strong foundation** for structured recipe extraction. The model achieves **high accuracy** and can be further improved with deeper context modeling and additional data.

