
# Cross-validation pipeline

## Diagram

![cv pipeline diagram](figs/cv_pipeline.png)

## In words 
1. 90%/10% Learning/Validation split
    - Each set has brain, behavioural, and confound data
2. Preprocessing:
    - Fit preprocessor to Learning set
        - Confound data
            - Replace missing data with median (`sklearn`'s `SimpleImputer`)
            - Inverse normal transformation (`sklearn`'s `QuantileTransformer`)
            - Z-score (`sklearn`'s `StandardScaler`)
        - Brain/behavioural data
            - Inverse normal transformation (`sklearn`'s `QuantileTransformer`)
            - Z-score (`sklearn`'s `StandardScaler`)
            - Deconfound using preprocessed confound data
            - PCA (with missing data)
                - Compute covariance matrix (`n_features`-by-`n_features`)
                - Project to nearest symmetric positive definite matrix
                - Do PCA via eigendecomposition
    - Apply fitted preprocessor to Validation set
3. Repeated cross-validation (repeat 500 times):
    - Split Learning set randomly into 5 folds
        - Train set: 4 folds
        - Test set: remaining fold
    - Fit CCA model on Train set
        - Get transformation weights (PCs x CAs)
        - *Previous approach: apply weights to Test set, get canonical scores (Test subjects x CAs)*
    - Repeat 5 times, each time using a different fold as Test set
        - *Previous approach: combine all Test scores to form a complete set of scores (all subjects)*
4. Combine results over all repetitions
    - Take mean of weights, then do SVD (as described [here](https://stackoverflow.com/questions/51517466/what-is-the-correct-way-to-average-several-rotation-matrices))
    - *Previous approach: take central tendency (mean/median) of score distributions*
        - *Problem: how to get equivalent measure for Validation set?*
            - *Would need to get weights back from Learning set scores*
5. Evaluation
    - Use cross-validated weights to get final Learning/Validation scores
    - Compare:
        - Canonical correlations
        - Regression with age
