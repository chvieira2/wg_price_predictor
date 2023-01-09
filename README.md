## What is the data?
WG ads in wg-gesucht.de are collect by the [housing_crawler](https://github.com/chvieira2/housing_crawler) app since August 2022. With roughly 750 new offers posted everyday, this dataset is fast growing. Collected data include information on the size and type of WG and on physical characteristics of the flatshare.
On top of that, I incorporate to the set of data collected per WG the life quality measurements produced by [livablestreets](https://github.com/chvieira2/livablestreets). For every address, I add information on surrounding city objects like streets, banks, stores, parks, lakes, etc.

So far, the dataset comprises >50.000 entries and >150 features.


## Creating a predictive model
The predictive model was created using sklearn's Pipeline module. The creation of the predictive model pipeline consists of three steps:
1. First, I encoded categorical features and transformed and scaled numerical features with PowerScaling and either MinMax or Standard scalling.
2. Next, I identified the most relevant numerical features in three ways:
- **Minimizing**: Numeric features with too little information were removed from analysis. This is meant to improve prediction by reducing dimentionality, as numerical features with identical values in all entries carry little to no information.
- **Variance Inflation Factor (VIF)**: Analysis of VIF was used to identify multi-colliniarity between features. Features with VIF higher than 10 were systematically excluded.
- **Permutation importance**: Analysis of feature importance by permutation was used to identify features with significant impact on the model. Features with low importance (<0.001) were systematically excluded.
3. Several regression models were automatically searched and cross-validated with GridSearchCV. Tested models include regularized linear models, neighbors clustering models, support vector machine models, bagged and stacked decision trees models, and a neural network. The best version of each model was KFold validated and the best scoring method selected.

Finally, the best model was included into the processing pipeline to create the final predictive model pipeline that has been trained in the whole dataset. 
