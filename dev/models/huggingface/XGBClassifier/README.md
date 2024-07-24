---
library_name: sklearn
license: mit
tags:
- sklearn
- skops
- tabular-classification
model_format: pickle
model_file: XGBClassifier.joblib
widget:
- structuredData:
    age:
    - 50
    - 31
    - 32
    bd2:
    - 0.627
    - 0.351
    - 0.672
    id:
    - ICU200010
    - ICU200011
    - ICU200012
    insurance:
    - 0
    - 0
    - 1
    m11:
    - 33.6
    - 26.6
    - 23.3
    pl:
    - 148
    - 85
    - 183
    pr:
    - 72
    - 66
    - 64
    prg:
    - 6
    - 1
    - 8
    sepsis:
    - Positive
    - Negative
    - Positive
    sk:
    - 35
    - 29
    - 0
    ts:
    - 0
    - 0
    - 0
---

# Model description

[More Information Needed]

## Intended uses & limitations

[More Information Needed]

## Training Procedure

[More Information Needed]

### Hyperparameters

<details>
<summary> Click to expand </summary>

| Hyperparameter                                                               | Value                                                                                                                                           |
|------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| memory                                                                       |                                                                                                                                                 |
| steps                                                                        | [('preprocessor', ColumnTransformer(transformers=[('numerical_pipeline',<br />                                 Pipeline(steps=[('log_transformations',<br />                                                  FunctionTransformer(func=<ufunc 'log1p'>)),<br />                                                 ('imputer',<br />                                                  SimpleImputer(strategy='median')),<br />                                                 ('scaler', RobustScaler())]),<br />                                 ['prg', 'pl', 'pr', 'sk', 'ts', 'm11', 'bd2',<br />                                  'age']),<br />                                ('categorical_pipeline',<br />                                 Pipeline(steps=[('as_categorical',<br />                                                  FunctionTransformer(func=<function as_...<br />                                                                handle_unknown='infrequent_if_exist',<br />                                                                sparse_output=False))]),<br />                                 ['insurance']),<br />                                ('feature_creation_pipeline',<br />                                 Pipeline(steps=[('feature_creation',<br />                                                  FunctionTransformer(func=<function feature_creation at 0x00000147012327A0>)),<br />                                                 ('imputer',<br />                                                  SimpleImputer(strategy='most_frequent')),<br />                                                 ('encoder',<br />                                                  OneHotEncoder(drop='first',<br />                                                                handle_unknown='infrequent_if_exist',<br />                                                                sparse_output=False))]),<br />                                 ['age'])])), ('feature-selection', SelectKBest(k='all',<br />            score_func=<function mutual_info_classif at 0x000001470129BA60>)), ('classifier', XGBClassifier(base_score=None, booster=None, callbacks=None,<br />              colsample_bylevel=None, colsample_bynode=None,<br />              colsample_bytree=None, device=None, early_stopping_rounds=None,<br />              enable_categorical=False, eval_metric=None, feature_types=None,<br />              gamma=None, grow_policy=None, importance_type=None,<br />              interaction_constraints=None, learning_rate=None, max_bin=None,<br />              max_cat_threshold=None, max_cat_to_onehot=None,<br />              max_delta_step=None, max_depth=20, max_leaves=None,<br />              min_child_weight=None, missing=nan, monotone_constraints=None,<br />              multi_strategy=None, n_estimators=10, n_jobs=-1,<br />              num_parallel_tree=None, random_state=2024, ...))]                                                                                                                                                 |
| verbose                                                                      | False                                                                                                                                           |
| preprocessor                                                                 | ColumnTransformer(transformers=[('numerical_pipeline',<br />                                 Pipeline(steps=[('log_transformations',<br />                                                  FunctionTransformer(func=<ufunc 'log1p'>)),<br />                                                 ('imputer',<br />                                                  SimpleImputer(strategy='median')),<br />                                                 ('scaler', RobustScaler())]),<br />                                 ['prg', 'pl', 'pr', 'sk', 'ts', 'm11', 'bd2',<br />                                  'age']),<br />                                ('categorical_pipeline',<br />                                 Pipeline(steps=[('as_categorical',<br />                                                  FunctionTransformer(func=<function as_...<br />                                                                handle_unknown='infrequent_if_exist',<br />                                                                sparse_output=False))]),<br />                                 ['insurance']),<br />                                ('feature_creation_pipeline',<br />                                 Pipeline(steps=[('feature_creation',<br />                                                  FunctionTransformer(func=<function feature_creation at 0x00000147012327A0>)),<br />                                                 ('imputer',<br />                                                  SimpleImputer(strategy='most_frequent')),<br />                                                 ('encoder',<br />                                                  OneHotEncoder(drop='first',<br />                                                                handle_unknown='infrequent_if_exist',<br />                                                                sparse_output=False))]),<br />                                 ['age'])])                                                                                                                                                 |
| feature-selection                                                            | SelectKBest(k='all',<br />            score_func=<function mutual_info_classif at 0x000001470129BA60>)                                                                                                                                                 |
| classifier                                                                   | XGBClassifier(base_score=None, booster=None, callbacks=None,<br />              colsample_bylevel=None, colsample_bynode=None,<br />              colsample_bytree=None, device=None, early_stopping_rounds=None,<br />              enable_categorical=False, eval_metric=None, feature_types=None,<br />              gamma=None, grow_policy=None, importance_type=None,<br />              interaction_constraints=None, learning_rate=None, max_bin=None,<br />              max_cat_threshold=None, max_cat_to_onehot=None,<br />              max_delta_step=None, max_depth=20, max_leaves=None,<br />              min_child_weight=None, missing=nan, monotone_constraints=None,<br />              multi_strategy=None, n_estimators=10, n_jobs=-1,<br />              num_parallel_tree=None, random_state=2024, ...)                                                                                                                                                 |
| preprocessor__force_int_remainder_cols                                       | True                                                                                                                                            |
| preprocessor__n_jobs                                                         |                                                                                                                                                 |
| preprocessor__remainder                                                      | drop                                                                                                                                            |
| preprocessor__sparse_threshold                                               | 0.3                                                                                                                                             |
| preprocessor__transformer_weights                                            |                                                                                                                                                 |
| preprocessor__transformers                                                   | [('numerical_pipeline', Pipeline(steps=[('log_transformations',<br />                 FunctionTransformer(func=<ufunc 'log1p'>)),<br />                ('imputer', SimpleImputer(strategy='median')),<br />                ('scaler', RobustScaler())]), ['prg', 'pl', 'pr', 'sk', 'ts', 'm11', 'bd2', 'age']), ('categorical_pipeline', Pipeline(steps=[('as_categorical',<br />                 FunctionTransformer(func=<function as_category at 0x0000014701232160>)),<br />                ('imputer', SimpleImputer(strategy='most_frequent')),<br />                ('encoder',<br />                 OneHotEncoder(drop='first',<br />                               handle_unknown='infrequent_if_exist',<br />                               sparse_output=False))]), ['insurance']), ('feature_creation_pipeline', Pipeline(steps=[('feature_creation',<br />                 FunctionTransformer(func=<function feature_creation at 0x00000147012327A0>)),<br />                ('imputer', SimpleImputer(strategy='most_frequent')),<br />                ('encoder',<br />                 OneHotEncoder(drop='first',<br />                               handle_unknown='infrequent_if_exist',<br />                               sparse_output=False))]), ['age'])]                                                                                                                                                 |
| preprocessor__verbose                                                        | False                                                                                                                                           |
| preprocessor__verbose_feature_names_out                                      | True                                                                                                                                            |
| preprocessor__numerical_pipeline                                             | Pipeline(steps=[('log_transformations',<br />                 FunctionTransformer(func=<ufunc 'log1p'>)),<br />                ('imputer', SimpleImputer(strategy='median')),<br />                ('scaler', RobustScaler())])                                                                                                                                                 |
| preprocessor__categorical_pipeline                                           | Pipeline(steps=[('as_categorical',<br />                 FunctionTransformer(func=<function as_category at 0x0000014701232160>)),<br />                ('imputer', SimpleImputer(strategy='most_frequent')),<br />                ('encoder',<br />                 OneHotEncoder(drop='first',<br />                               handle_unknown='infrequent_if_exist',<br />                               sparse_output=False))])                                                                                                                                                 |
| preprocessor__feature_creation_pipeline                                      | Pipeline(steps=[('feature_creation',<br />                 FunctionTransformer(func=<function feature_creation at 0x00000147012327A0>)),<br />                ('imputer', SimpleImputer(strategy='most_frequent')),<br />                ('encoder',<br />                 OneHotEncoder(drop='first',<br />                               handle_unknown='infrequent_if_exist',<br />                               sparse_output=False))])                                                                                                                                                 |
| preprocessor__numerical_pipeline__memory                                     |                                                                                                                                                 |
| preprocessor__numerical_pipeline__steps                                      | [('log_transformations', FunctionTransformer(func=<ufunc 'log1p'>)), ('imputer', SimpleImputer(strategy='median')), ('scaler', RobustScaler())] |
| preprocessor__numerical_pipeline__verbose                                    | False                                                                                                                                           |
| preprocessor__numerical_pipeline__log_transformations                        | FunctionTransformer(func=<ufunc 'log1p'>)                                                                                                       |
| preprocessor__numerical_pipeline__imputer                                    | SimpleImputer(strategy='median')                                                                                                                |
| preprocessor__numerical_pipeline__scaler                                     | RobustScaler()                                                                                                                                  |
| preprocessor__numerical_pipeline__log_transformations__accept_sparse         | False                                                                                                                                           |
| preprocessor__numerical_pipeline__log_transformations__check_inverse         | True                                                                                                                                            |
| preprocessor__numerical_pipeline__log_transformations__feature_names_out     |                                                                                                                                                 |
| preprocessor__numerical_pipeline__log_transformations__func                  | <ufunc 'log1p'>                                                                                                                                 |
| preprocessor__numerical_pipeline__log_transformations__inv_kw_args           |                                                                                                                                                 |
| preprocessor__numerical_pipeline__log_transformations__inverse_func          |                                                                                                                                                 |
| preprocessor__numerical_pipeline__log_transformations__kw_args               |                                                                                                                                                 |
| preprocessor__numerical_pipeline__log_transformations__validate              | False                                                                                                                                           |
| preprocessor__numerical_pipeline__imputer__add_indicator                     | False                                                                                                                                           |
| preprocessor__numerical_pipeline__imputer__copy                              | True                                                                                                                                            |
| preprocessor__numerical_pipeline__imputer__fill_value                        |                                                                                                                                                 |
| preprocessor__numerical_pipeline__imputer__keep_empty_features               | False                                                                                                                                           |
| preprocessor__numerical_pipeline__imputer__missing_values                    | nan                                                                                                                                             |
| preprocessor__numerical_pipeline__imputer__strategy                          | median                                                                                                                                          |
| preprocessor__numerical_pipeline__scaler__copy                               | True                                                                                                                                            |
| preprocessor__numerical_pipeline__scaler__quantile_range                     | (25.0, 75.0)                                                                                                                                    |
| preprocessor__numerical_pipeline__scaler__unit_variance                      | False                                                                                                                                           |
| preprocessor__numerical_pipeline__scaler__with_centering                     | True                                                                                                                                            |
| preprocessor__numerical_pipeline__scaler__with_scaling                       | True                                                                                                                                            |
| preprocessor__categorical_pipeline__memory                                   |                                                                                                                                                 |
| preprocessor__categorical_pipeline__steps                                    | [('as_categorical', FunctionTransformer(func=<function as_category at 0x0000014701232160>)), ('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(drop='first', handle_unknown='infrequent_if_exist',<br />              sparse_output=False))]                                                                                                                                                 |
| preprocessor__categorical_pipeline__verbose                                  | False                                                                                                                                           |
| preprocessor__categorical_pipeline__as_categorical                           | FunctionTransformer(func=<function as_category at 0x0000014701232160>)                                                                          |
| preprocessor__categorical_pipeline__imputer                                  | SimpleImputer(strategy='most_frequent')                                                                                                         |
| preprocessor__categorical_pipeline__encoder                                  | OneHotEncoder(drop='first', handle_unknown='infrequent_if_exist',<br />              sparse_output=False)                                                                                                                                                 |
| preprocessor__categorical_pipeline__as_categorical__accept_sparse            | False                                                                                                                                           |
| preprocessor__categorical_pipeline__as_categorical__check_inverse            | True                                                                                                                                            |
| preprocessor__categorical_pipeline__as_categorical__feature_names_out        |                                                                                                                                                 |
| preprocessor__categorical_pipeline__as_categorical__func                     | <function as_category at 0x0000014701232160>                                                                                                    |
| preprocessor__categorical_pipeline__as_categorical__inv_kw_args              |                                                                                                                                                 |
| preprocessor__categorical_pipeline__as_categorical__inverse_func             |                                                                                                                                                 |
| preprocessor__categorical_pipeline__as_categorical__kw_args                  |                                                                                                                                                 |
| preprocessor__categorical_pipeline__as_categorical__validate                 | False                                                                                                                                           |
| preprocessor__categorical_pipeline__imputer__add_indicator                   | False                                                                                                                                           |
| preprocessor__categorical_pipeline__imputer__copy                            | True                                                                                                                                            |
| preprocessor__categorical_pipeline__imputer__fill_value                      |                                                                                                                                                 |
| preprocessor__categorical_pipeline__imputer__keep_empty_features             | False                                                                                                                                           |
| preprocessor__categorical_pipeline__imputer__missing_values                  | nan                                                                                                                                             |
| preprocessor__categorical_pipeline__imputer__strategy                        | most_frequent                                                                                                                                   |
| preprocessor__categorical_pipeline__encoder__categories                      | auto                                                                                                                                            |
| preprocessor__categorical_pipeline__encoder__drop                            | first                                                                                                                                           |
| preprocessor__categorical_pipeline__encoder__dtype                           | <class 'numpy.float64'>                                                                                                                         |
| preprocessor__categorical_pipeline__encoder__feature_name_combiner           | concat                                                                                                                                          |
| preprocessor__categorical_pipeline__encoder__handle_unknown                  | infrequent_if_exist                                                                                                                             |
| preprocessor__categorical_pipeline__encoder__max_categories                  |                                                                                                                                                 |
| preprocessor__categorical_pipeline__encoder__min_frequency                   |                                                                                                                                                 |
| preprocessor__categorical_pipeline__encoder__sparse_output                   | False                                                                                                                                           |
| preprocessor__feature_creation_pipeline__memory                              |                                                                                                                                                 |
| preprocessor__feature_creation_pipeline__steps                               | [('feature_creation', FunctionTransformer(func=<function feature_creation at 0x00000147012327A0>)), ('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(drop='first', handle_unknown='infrequent_if_exist',<br />              sparse_output=False))]                                                                                                                                                 |
| preprocessor__feature_creation_pipeline__verbose                             | False                                                                                                                                           |
| preprocessor__feature_creation_pipeline__feature_creation                    | FunctionTransformer(func=<function feature_creation at 0x00000147012327A0>)                                                                     |
| preprocessor__feature_creation_pipeline__imputer                             | SimpleImputer(strategy='most_frequent')                                                                                                         |
| preprocessor__feature_creation_pipeline__encoder                             | OneHotEncoder(drop='first', handle_unknown='infrequent_if_exist',<br />              sparse_output=False)                                                                                                                                                 |
| preprocessor__feature_creation_pipeline__feature_creation__accept_sparse     | False                                                                                                                                           |
| preprocessor__feature_creation_pipeline__feature_creation__check_inverse     | True                                                                                                                                            |
| preprocessor__feature_creation_pipeline__feature_creation__feature_names_out |                                                                                                                                                 |
| preprocessor__feature_creation_pipeline__feature_creation__func              | <function feature_creation at 0x00000147012327A0>                                                                                               |
| preprocessor__feature_creation_pipeline__feature_creation__inv_kw_args       |                                                                                                                                                 |
| preprocessor__feature_creation_pipeline__feature_creation__inverse_func      |                                                                                                                                                 |
| preprocessor__feature_creation_pipeline__feature_creation__kw_args           |                                                                                                                                                 |
| preprocessor__feature_creation_pipeline__feature_creation__validate          | False                                                                                                                                           |
| preprocessor__feature_creation_pipeline__imputer__add_indicator              | False                                                                                                                                           |
| preprocessor__feature_creation_pipeline__imputer__copy                       | True                                                                                                                                            |
| preprocessor__feature_creation_pipeline__imputer__fill_value                 |                                                                                                                                                 |
| preprocessor__feature_creation_pipeline__imputer__keep_empty_features        | False                                                                                                                                           |
| preprocessor__feature_creation_pipeline__imputer__missing_values             | nan                                                                                                                                             |
| preprocessor__feature_creation_pipeline__imputer__strategy                   | most_frequent                                                                                                                                   |
| preprocessor__feature_creation_pipeline__encoder__categories                 | auto                                                                                                                                            |
| preprocessor__feature_creation_pipeline__encoder__drop                       | first                                                                                                                                           |
| preprocessor__feature_creation_pipeline__encoder__dtype                      | <class 'numpy.float64'>                                                                                                                         |
| preprocessor__feature_creation_pipeline__encoder__feature_name_combiner      | concat                                                                                                                                          |
| preprocessor__feature_creation_pipeline__encoder__handle_unknown             | infrequent_if_exist                                                                                                                             |
| preprocessor__feature_creation_pipeline__encoder__max_categories             |                                                                                                                                                 |
| preprocessor__feature_creation_pipeline__encoder__min_frequency              |                                                                                                                                                 |
| preprocessor__feature_creation_pipeline__encoder__sparse_output              | False                                                                                                                                           |
| feature-selection__k                                                         | all                                                                                                                                             |
| feature-selection__score_func                                                | <function mutual_info_classif at 0x000001470129BA60>                                                                                            |
| classifier__objective                                                        | binary:logistic                                                                                                                                 |
| classifier__base_score                                                       |                                                                                                                                                 |
| classifier__booster                                                          |                                                                                                                                                 |
| classifier__callbacks                                                        |                                                                                                                                                 |
| classifier__colsample_bylevel                                                |                                                                                                                                                 |
| classifier__colsample_bynode                                                 |                                                                                                                                                 |
| classifier__colsample_bytree                                                 |                                                                                                                                                 |
| classifier__device                                                           |                                                                                                                                                 |
| classifier__early_stopping_rounds                                            |                                                                                                                                                 |
| classifier__enable_categorical                                               | False                                                                                                                                           |
| classifier__eval_metric                                                      |                                                                                                                                                 |
| classifier__feature_types                                                    |                                                                                                                                                 |
| classifier__gamma                                                            |                                                                                                                                                 |
| classifier__grow_policy                                                      |                                                                                                                                                 |
| classifier__importance_type                                                  |                                                                                                                                                 |
| classifier__interaction_constraints                                          |                                                                                                                                                 |
| classifier__learning_rate                                                    |                                                                                                                                                 |
| classifier__max_bin                                                          |                                                                                                                                                 |
| classifier__max_cat_threshold                                                |                                                                                                                                                 |
| classifier__max_cat_to_onehot                                                |                                                                                                                                                 |
| classifier__max_delta_step                                                   |                                                                                                                                                 |
| classifier__max_depth                                                        | 20                                                                                                                                              |
| classifier__max_leaves                                                       |                                                                                                                                                 |
| classifier__min_child_weight                                                 |                                                                                                                                                 |
| classifier__missing                                                          | nan                                                                                                                                             |
| classifier__monotone_constraints                                             |                                                                                                                                                 |
| classifier__multi_strategy                                                   |                                                                                                                                                 |
| classifier__n_estimators                                                     | 10                                                                                                                                              |
| classifier__n_jobs                                                           | -1                                                                                                                                              |
| classifier__num_parallel_tree                                                |                                                                                                                                                 |
| classifier__random_state                                                     | 2024                                                                                                                                            |
| classifier__reg_alpha                                                        |                                                                                                                                                 |
| classifier__reg_lambda                                                       |                                                                                                                                                 |
| classifier__sampling_method                                                  |                                                                                                                                                 |
| classifier__scale_pos_weight                                                 |                                                                                                                                                 |
| classifier__subsample                                                        |                                                                                                                                                 |
| classifier__tree_method                                                      |                                                                                                                                                 |
| classifier__validate_parameters                                              |                                                                                                                                                 |
| classifier__verbosity                                                        |                                                                                                                                                 |
| classifier__verbose                                                          | 0                                                                                                                                               |

</details>

### Model Plot

<style>#sk-container-id-8 {/* Definition of color scheme common for light and dark mode */--sklearn-color-text: black;--sklearn-color-line: gray;/* Definition of color scheme for unfitted estimators */--sklearn-color-unfitted-level-0: #fff5e6;--sklearn-color-unfitted-level-1: #f6e4d2;--sklearn-color-unfitted-level-2: #ffe0b3;--sklearn-color-unfitted-level-3: chocolate;/* Definition of color scheme for fitted estimators */--sklearn-color-fitted-level-0: #f0f8ff;--sklearn-color-fitted-level-1: #d4ebff;--sklearn-color-fitted-level-2: #b3dbfd;--sklearn-color-fitted-level-3: cornflowerblue;/* Specific color for light theme */--sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));--sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));--sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));--sklearn-color-icon: #696969;@media (prefers-color-scheme: dark) {/* Redefinition of color scheme for dark theme */--sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));--sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));--sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));--sklearn-color-icon: #878787;}
}#sk-container-id-8 {color: var(--sklearn-color-text);
}#sk-container-id-8 pre {padding: 0;
}#sk-container-id-8 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;
}#sk-container-id-8 div.sk-dashed-wrapped {border: 1px dashed var(--sklearn-color-line);margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: var(--sklearn-color-background);
}#sk-container-id-8 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }`but bootstrap.min.css set `[hidden] { display: none !important; }`so we also need the `!important` here to be able to override thedefault hidden behavior on the sphinx rendered scikit-learn.org.See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;
}#sk-container-id-8 div.sk-text-repr-fallback {display: none;
}div.sk-parallel-item,
div.sk-serial,
div.sk-item {/* draw centered vertical line to link estimators */background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));background-size: 2px 100%;background-repeat: no-repeat;background-position: center center;
}/* Parallel-specific style estimator block */#sk-container-id-8 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 2px solid var(--sklearn-color-text-on-default-background);flex-grow: 1;
}#sk-container-id-8 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: var(--sklearn-color-background);position: relative;
}#sk-container-id-8 div.sk-parallel-item {display: flex;flex-direction: column;
}#sk-container-id-8 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;
}#sk-container-id-8 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;
}#sk-container-id-8 div.sk-parallel-item:only-child::after {width: 0;
}/* Serial-specific style estimator block */#sk-container-id-8 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: var(--sklearn-color-background);padding-right: 1em;padding-left: 1em;
}/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*//* Pipeline and ColumnTransformer style (default) */#sk-container-id-8 div.sk-toggleable {/* Default theme specific background. It is overwritten whether we have aspecific estimator or a Pipeline/ColumnTransformer */background-color: var(--sklearn-color-background);
}/* Toggleable label */
#sk-container-id-8 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.5em;box-sizing: border-box;text-align: center;
}#sk-container-id-8 label.sk-toggleable__label-arrow:before {/* Arrow on the left of the label */content: "▸";float: left;margin-right: 0.25em;color: var(--sklearn-color-icon);
}#sk-container-id-8 label.sk-toggleable__label-arrow:hover:before {color: var(--sklearn-color-text);
}/* Toggleable content - dropdown */#sk-container-id-8 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;/* unfitted */background-color: var(--sklearn-color-unfitted-level-0);
}#sk-container-id-8 div.sk-toggleable__content.fitted {/* fitted */background-color: var(--sklearn-color-fitted-level-0);
}#sk-container-id-8 div.sk-toggleable__content pre {margin: 0.2em;border-radius: 0.25em;color: var(--sklearn-color-text);/* unfitted */background-color: var(--sklearn-color-unfitted-level-0);
}#sk-container-id-8 div.sk-toggleable__content.fitted pre {/* unfitted */background-color: var(--sklearn-color-fitted-level-0);
}#sk-container-id-8 input.sk-toggleable__control:checked~div.sk-toggleable__content {/* Expand drop-down */max-height: 200px;max-width: 100%;overflow: auto;
}#sk-container-id-8 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";
}/* Pipeline/ColumnTransformer-specific style */#sk-container-id-8 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {color: var(--sklearn-color-text);background-color: var(--sklearn-color-unfitted-level-2);
}#sk-container-id-8 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: var(--sklearn-color-fitted-level-2);
}/* Estimator-specific style *//* Colorize estimator box */
#sk-container-id-8 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {/* unfitted */background-color: var(--sklearn-color-unfitted-level-2);
}#sk-container-id-8 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {/* fitted */background-color: var(--sklearn-color-fitted-level-2);
}#sk-container-id-8 div.sk-label label.sk-toggleable__label,
#sk-container-id-8 div.sk-label label {/* The background is the default theme color */color: var(--sklearn-color-text-on-default-background);
}/* On hover, darken the color of the background */
#sk-container-id-8 div.sk-label:hover label.sk-toggleable__label {color: var(--sklearn-color-text);background-color: var(--sklearn-color-unfitted-level-2);
}/* Label box, darken color on hover, fitted */
#sk-container-id-8 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {color: var(--sklearn-color-text);background-color: var(--sklearn-color-fitted-level-2);
}/* Estimator label */#sk-container-id-8 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;
}#sk-container-id-8 div.sk-label-container {text-align: center;
}/* Estimator-specific */
#sk-container-id-8 div.sk-estimator {font-family: monospace;border: 1px dotted var(--sklearn-color-border-box);border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;/* unfitted */background-color: var(--sklearn-color-unfitted-level-0);
}#sk-container-id-8 div.sk-estimator.fitted {/* fitted */background-color: var(--sklearn-color-fitted-level-0);
}/* on hover */
#sk-container-id-8 div.sk-estimator:hover {/* unfitted */background-color: var(--sklearn-color-unfitted-level-2);
}#sk-container-id-8 div.sk-estimator.fitted:hover {/* fitted */background-color: var(--sklearn-color-fitted-level-2);
}/* Specification for estimator info (e.g. "i" and "?") *//* Common style for "i" and "?" */.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {float: right;font-size: smaller;line-height: 1em;font-family: monospace;background-color: var(--sklearn-color-background);border-radius: 1em;height: 1em;width: 1em;text-decoration: none !important;margin-left: 1ex;/* unfitted */border: var(--sklearn-color-unfitted-level-1) 1pt solid;color: var(--sklearn-color-unfitted-level-1);
}.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {/* fitted */border: var(--sklearn-color-fitted-level-1) 1pt solid;color: var(--sklearn-color-fitted-level-1);
}/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {/* unfitted */background-color: var(--sklearn-color-unfitted-level-3);color: var(--sklearn-color-background);text-decoration: none;
}div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {/* fitted */background-color: var(--sklearn-color-fitted-level-3);color: var(--sklearn-color-background);text-decoration: none;
}/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {display: none;z-index: 9999;position: relative;font-weight: normal;right: .2ex;padding: .5ex;margin: .5ex;width: min-content;min-width: 20ex;max-width: 50ex;color: var(--sklearn-color-text);box-shadow: 2pt 2pt 4pt #999;/* unfitted */background: var(--sklearn-color-unfitted-level-0);border: .5pt solid var(--sklearn-color-unfitted-level-3);
}.sk-estimator-doc-link.fitted span {/* fitted */background: var(--sklearn-color-fitted-level-0);border: var(--sklearn-color-fitted-level-3);
}.sk-estimator-doc-link:hover span {display: block;
}/* "?"-specific style due to the `<a>` HTML tag */#sk-container-id-8 a.estimator_doc_link {float: right;font-size: 1rem;line-height: 1em;font-family: monospace;background-color: var(--sklearn-color-background);border-radius: 1rem;height: 1rem;width: 1rem;text-decoration: none;/* unfitted */color: var(--sklearn-color-unfitted-level-1);border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}#sk-container-id-8 a.estimator_doc_link.fitted {/* fitted */border: var(--sklearn-color-fitted-level-1) 1pt solid;color: var(--sklearn-color-fitted-level-1);
}/* On hover */
#sk-container-id-8 a.estimator_doc_link:hover {/* unfitted */background-color: var(--sklearn-color-unfitted-level-3);color: var(--sklearn-color-background);text-decoration: none;
}#sk-container-id-8 a.estimator_doc_link.fitted:hover {/* fitted */background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-8" class="sk-top-container" style="overflow: auto;"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;preprocessor&#x27;,ColumnTransformer(transformers=[(&#x27;numerical_pipeline&#x27;,Pipeline(steps=[(&#x27;log_transformations&#x27;,FunctionTransformer(func=&lt;ufunc &#x27;log1p&#x27;&gt;)),(&#x27;imputer&#x27;,SimpleImputer(strategy=&#x27;median&#x27;)),(&#x27;scaler&#x27;,RobustScaler())]),[&#x27;prg&#x27;, &#x27;pl&#x27;, &#x27;pr&#x27;, &#x27;sk&#x27;,&#x27;ts&#x27;, &#x27;m11&#x27;, &#x27;bd2&#x27;, &#x27;age&#x27;]),(&#x27;categorical_pipeline&#x27;,Pipeline(steps=[(&#x27;as_categorical&#x27;,Funct...feature_types=None, gamma=None, grow_policy=None,importance_type=None,interaction_constraints=None, learning_rate=None,max_bin=None, max_cat_threshold=None,max_cat_to_onehot=None, max_delta_step=None,max_depth=20, max_leaves=None,min_child_weight=None, missing=nan,monotone_constraints=None, multi_strategy=None,n_estimators=10, n_jobs=-1,num_parallel_tree=None, random_state=2024, ...))])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-74" type="checkbox" ><label for="sk-estimator-id-74" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;Pipeline<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.pipeline.Pipeline.html">?<span>Documentation for Pipeline</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>Pipeline(steps=[(&#x27;preprocessor&#x27;,ColumnTransformer(transformers=[(&#x27;numerical_pipeline&#x27;,Pipeline(steps=[(&#x27;log_transformations&#x27;,FunctionTransformer(func=&lt;ufunc &#x27;log1p&#x27;&gt;)),(&#x27;imputer&#x27;,SimpleImputer(strategy=&#x27;median&#x27;)),(&#x27;scaler&#x27;,RobustScaler())]),[&#x27;prg&#x27;, &#x27;pl&#x27;, &#x27;pr&#x27;, &#x27;sk&#x27;,&#x27;ts&#x27;, &#x27;m11&#x27;, &#x27;bd2&#x27;, &#x27;age&#x27;]),(&#x27;categorical_pipeline&#x27;,Pipeline(steps=[(&#x27;as_categorical&#x27;,Funct...feature_types=None, gamma=None, grow_policy=None,importance_type=None,interaction_constraints=None, learning_rate=None,max_bin=None, max_cat_threshold=None,max_cat_to_onehot=None, max_delta_step=None,max_depth=20, max_leaves=None,min_child_weight=None, missing=nan,monotone_constraints=None, multi_strategy=None,n_estimators=10, n_jobs=-1,num_parallel_tree=None, random_state=2024, ...))])</pre></div> </div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-75" type="checkbox" ><label for="sk-estimator-id-75" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;preprocessor: ColumnTransformer<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.compose.ColumnTransformer.html">?<span>Documentation for preprocessor: ColumnTransformer</span></a></label><div class="sk-toggleable__content fitted"><pre>ColumnTransformer(transformers=[(&#x27;numerical_pipeline&#x27;,Pipeline(steps=[(&#x27;log_transformations&#x27;,FunctionTransformer(func=&lt;ufunc &#x27;log1p&#x27;&gt;)),(&#x27;imputer&#x27;,SimpleImputer(strategy=&#x27;median&#x27;)),(&#x27;scaler&#x27;, RobustScaler())]),[&#x27;prg&#x27;, &#x27;pl&#x27;, &#x27;pr&#x27;, &#x27;sk&#x27;, &#x27;ts&#x27;, &#x27;m11&#x27;, &#x27;bd2&#x27;,&#x27;age&#x27;]),(&#x27;categorical_pipeline&#x27;,Pipeline(steps=[(&#x27;as_categorical&#x27;,FunctionTransformer(func=&lt;function as_...handle_unknown=&#x27;infrequent_if_exist&#x27;,sparse_output=False))]),[&#x27;insurance&#x27;]),(&#x27;feature_creation_pipeline&#x27;,Pipeline(steps=[(&#x27;feature_creation&#x27;,FunctionTransformer(func=&lt;function feature_creation at 0x00000147012327A0&gt;)),(&#x27;imputer&#x27;,SimpleImputer(strategy=&#x27;most_frequent&#x27;)),(&#x27;encoder&#x27;,OneHotEncoder(drop=&#x27;first&#x27;,handle_unknown=&#x27;infrequent_if_exist&#x27;,sparse_output=False))]),[&#x27;age&#x27;])])</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-76" type="checkbox" ><label for="sk-estimator-id-76" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">numerical_pipeline</label><div class="sk-toggleable__content fitted"><pre>[&#x27;prg&#x27;, &#x27;pl&#x27;, &#x27;pr&#x27;, &#x27;sk&#x27;, &#x27;ts&#x27;, &#x27;m11&#x27;, &#x27;bd2&#x27;, &#x27;age&#x27;]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-77" type="checkbox" ><label for="sk-estimator-id-77" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;FunctionTransformer<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.FunctionTransformer.html">?<span>Documentation for FunctionTransformer</span></a></label><div class="sk-toggleable__content fitted"><pre>FunctionTransformer(func=&lt;ufunc &#x27;log1p&#x27;&gt;)</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-78" type="checkbox" ><label for="sk-estimator-id-78" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;SimpleImputer<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.impute.SimpleImputer.html">?<span>Documentation for SimpleImputer</span></a></label><div class="sk-toggleable__content fitted"><pre>SimpleImputer(strategy=&#x27;median&#x27;)</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-79" type="checkbox" ><label for="sk-estimator-id-79" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;RobustScaler<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.RobustScaler.html">?<span>Documentation for RobustScaler</span></a></label><div class="sk-toggleable__content fitted"><pre>RobustScaler()</pre></div> </div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-80" type="checkbox" ><label for="sk-estimator-id-80" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">categorical_pipeline</label><div class="sk-toggleable__content fitted"><pre>[&#x27;insurance&#x27;]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-81" type="checkbox" ><label for="sk-estimator-id-81" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;FunctionTransformer<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.FunctionTransformer.html">?<span>Documentation for FunctionTransformer</span></a></label><div class="sk-toggleable__content fitted"><pre>FunctionTransformer(func=&lt;function as_category at 0x0000014701232160&gt;)</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-82" type="checkbox" ><label for="sk-estimator-id-82" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;SimpleImputer<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.impute.SimpleImputer.html">?<span>Documentation for SimpleImputer</span></a></label><div class="sk-toggleable__content fitted"><pre>SimpleImputer(strategy=&#x27;most_frequent&#x27;)</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-83" type="checkbox" ><label for="sk-estimator-id-83" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;OneHotEncoder<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.OneHotEncoder.html">?<span>Documentation for OneHotEncoder</span></a></label><div class="sk-toggleable__content fitted"><pre>OneHotEncoder(drop=&#x27;first&#x27;, handle_unknown=&#x27;infrequent_if_exist&#x27;,sparse_output=False)</pre></div> </div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-84" type="checkbox" ><label for="sk-estimator-id-84" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">feature_creation_pipeline</label><div class="sk-toggleable__content fitted"><pre>[&#x27;age&#x27;]</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-85" type="checkbox" ><label for="sk-estimator-id-85" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;FunctionTransformer<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.FunctionTransformer.html">?<span>Documentation for FunctionTransformer</span></a></label><div class="sk-toggleable__content fitted"><pre>FunctionTransformer(func=&lt;function feature_creation at 0x00000147012327A0&gt;)</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-86" type="checkbox" ><label for="sk-estimator-id-86" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;SimpleImputer<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.impute.SimpleImputer.html">?<span>Documentation for SimpleImputer</span></a></label><div class="sk-toggleable__content fitted"><pre>SimpleImputer(strategy=&#x27;most_frequent&#x27;)</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-87" type="checkbox" ><label for="sk-estimator-id-87" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;OneHotEncoder<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.OneHotEncoder.html">?<span>Documentation for OneHotEncoder</span></a></label><div class="sk-toggleable__content fitted"><pre>OneHotEncoder(drop=&#x27;first&#x27;, handle_unknown=&#x27;infrequent_if_exist&#x27;,sparse_output=False)</pre></div> </div></div></div></div></div></div></div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-88" type="checkbox" ><label for="sk-estimator-id-88" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;SelectKBest<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.feature_selection.SelectKBest.html">?<span>Documentation for SelectKBest</span></a></label><div class="sk-toggleable__content fitted"><pre>SelectKBest(k=&#x27;all&#x27;,score_func=&lt;function mutual_info_classif at 0x000001470129BA60&gt;)</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-89" type="checkbox" ><label for="sk-estimator-id-89" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">XGBClassifier</label><div class="sk-toggleable__content fitted"><pre>XGBClassifier(base_score=None, booster=None, callbacks=None,colsample_bylevel=None, colsample_bynode=None,colsample_bytree=None, device=None, early_stopping_rounds=None,enable_categorical=False, eval_metric=None, feature_types=None,gamma=None, grow_policy=None, importance_type=None,interaction_constraints=None, learning_rate=None, max_bin=None,max_cat_threshold=None, max_cat_to_onehot=None,max_delta_step=None, max_depth=20, max_leaves=None,min_child_weight=None, missing=nan, monotone_constraints=None,multi_strategy=None, n_estimators=10, n_jobs=-1,num_parallel_tree=None, random_state=2024, ...)</pre></div> </div></div></div></div></div></div>

## Evaluation Results

[More Information Needed]

# How to Get Started with the Model

[More Information Needed]

# Model Card Authors

This model card is written by following authors:

[More Information Needed]

# Model Card Contact

You can contact the model card authors through following channels:
[More Information Needed]

# Citation

Below you can find information related to citation.

**BibTeX:**
```
[More Information Needed]
```

# citation_bibtex

bibtex
@inproceedings{...,year={2024}}

# get_started_code

import joblib 
 clf = joblib.load(../models/XGBClassifier.joblib)

# model_card_authors

Gabriel Okundaye

# limitations

This model needs further feature engineering to improve the f1 weighted score. Collaborate on with me here [GitHub](https://github.com/D0nG4667/sepsis_prediction_full_stack)

# model_description

This is a XGBClassifier model trained on Sepsis dataset from this [kaggle dataset](https://www.kaggle.com/datasets/chaunguynnghunh/sepsis/data).

# roc_auc_curve

![roc_auc_curve](ROC_AUC_Curve_for_RandomForestClassifier_and_XGBClassifier_(F1-Weighted_Scores__0.778_and_0.777_respectively).webp)

# feature_importances

![feature_importances](Feature_Importances-_XGBClassifier_(F1-Weighted_Scores__0.777).webp)