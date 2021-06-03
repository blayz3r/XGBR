# 0. Initialization

# 0.1 Automatic Package Installation if they are Missing
list.of.packages <- c("insuranceData", "xgboost", "dplyr", "caret", "GGally", "vip", "pdp")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

# 0.2 Import Libraries
library(insuranceData) #The data
library(xgboost) #For the Modeling
library(dplyr) #Data mangling
library(caret) #Data Preparation
library(GGally) #Data Exploration
library(vip) #Variable Importance
library(pdp) #Partial Dependency Plot





# 1. Data Preparation
# 1.1 Import Dataset
data(dataOhlsson)

# 1.2 Rename Variables to English
dataOhlssonEN <- 
	dataOhlsson %>%
	transmute( #Renaming
		Age = agarald,
		Sex = kon,
		Territory = zon,
		MCClass = mcklass, # Engine Power / Weight Ratio
		VehicleAge = fordald, 
		BonusClass = bonuskl,   #Bonus Malus: Each Year Claim Free: +1, Each Claim -2
		EarnedExposure = duration,
		NbClaims = antskad,
		IncurredCost = skadkost 
	) %>%
	filter(EarnedExposure > 0)

# 1.3 Preview the data
head(dataOhlssonEN)








# 2. Data Exploration
# 2.1 Dimension of the data: nrows x ncols
dataOhlssonEN %>% dim() 

# 2.2 Summary of the variables
dataOhlssonEN %>% summary()

# 2.3 Automatic visualization of the variables interaction
ggpairs(
	dataOhlssonEN %>%
		sample_n(2000) %>% #We only need a 
		select(-IncurredCost),
	lower = list(continuous = wrap("smooth", alpha = 0.1))
)









# 3. Model Preparation
# 3.1 Variables to include in the model
dummy_model <- 
	dummyVars(
		~
			Age +
			Sex +
			as.factor(Territory) + #Transform Territory from numeric to character
			MCClass + 
			VehicleAge +
			BonusClass,
		data = dataOhlssonEN
	)

# 3.2 Create matrix of explanatory variables
mat1 <- predict(dummy_model, newdata = dataOhlssonEN)

# 3.3 Compute the average frequency that will be used as base rate
avg_freq <- 
	with(dataOhlssonEN, sum(NbClaims) / sum(EarnedExposure))

# 3.4 Gather all information in the XGBoost format
mat2 <- 	 
  xgb.DMatrix(
    mat1, #Matrix of explanatory variables
    label = dataOhlssonEN$NbClaims / dataOhlssonEN$EarnedExposure, #Response Variable: the frequency of claims
	weight = dataOhlssonEN$EarnedExposure, #Weight Variable: Earned Exposure
	base_margin = rep(log(avg_freq), nrow(dataOhlssonEN)), #BaseRate
  )







########################################################################

# 4. Frequency Modeling with a Poisson distribution
# 4.1 Using Cross-Validation to find the optimal parameters
# We need to find the optimal eta (learning parameter) that will give us around 100 trees
set.seed(9370707) #Set random seed to have reproducible results
xgb.cv(
	data = mat2,
	objective = 'count:poisson', #Like in a glm Poisson, we'll model using a Poisson distribution
	nrounds = 250, #Number of trees before stopping the algo
	max_depth = 3, #Size of the trees
	eta = 0.3, #Learning speed rate
	nfold = 5, #Train on 4/5 of the data and validate on the rest
	early_stopping_rounds = 5 #Computation will stop if the validation score doesn't improve in 5 rounds
  ) #For more info on parameters, visit https://xgboost.readthedocs.io/en/latest/parameter.html
#Best iteration:                                                                    
#[38]    train-poisson-nloglik:0.052556+0.000354 test-poisson-nloglik:0.054717+0.001519  

# Iteration #2
set.seed(9370707) 
xgb.cv(
	data = mat2,
	objective = 'count:poisson',
	nrounds = 250, 
	max_depth = 3, 
	eta = 0.1, #Trying a smaller eta
	nfold = 5, 
	early_stopping_rounds = 5
  )
# Best iteration:                                                                    
# [101]   train-poisson-nloglik:0.052750+0.000373 test-poisson-nloglik:0.054728+0.001481    

# Iteration #3
set.seed(9370707) 
xgb.cv(
	data = mat2,
	objective = 'count:poisson',
	nrounds = 250, 
	max_depth = 2, #Trying smaller trees
	eta = 0.1,
	nfold = 5, 
	early_stopping_rounds = 5
  )
# Best iteration:                                                                    
# [194]   train-poisson-nloglik:0.053196+0.000358 test-poisson-nloglik:0.054545+0.001346   

# Iteration #4
set.seed(9370707) 
xgb.cv(
	data = mat2,
	objective = 'count:poisson',
	nrounds = 250, 
	max_depth = 4, #Trying bigger trees
	eta = 0.1,
	nfold = 5, 
	early_stopping_rounds = 5
  )
# Best iteration:                                                                    
# [101]   train-poisson-nloglik:0.051445+0.000381 test-poisson-nloglik:0.054863+0.001450  

# 4.2 Train the final model using Iteration #3's parameters
xgb.model1 <- 
  xgb.train(
	data = mat2,
	objective = 'count:poisson', 
	nrounds = 194, 
	max_depth = 2,
	eta = 0.1
  )






# 5. Model Interpretation
# 5.1 Prediction Analysis
dataOhlssonEN$preds <- predict(xgb.model1, newdata = mat2)

summary(dataOhlssonEN$preds)
qplot(preds, weight = EarnedExposure, data = dataOhlssonEN, geom = "histogram")
qplot(preds, weight = EarnedExposure, data = dataOhlssonEN, geom = "histogram", xlim = c(0, 0.05))

# 5.2 Variable Importance
vip(xgb.model1)

qplot(log(preds), geom = "histogram")

predict(xgb.model1, newdata = mat2) %>%
qplot(as.vector(), geom = "density")

# 5.3 Tree Visualization
xgb.plot.tree(model = xgb.model1, trees=0:2)


# 5.4 Partial Dependency Plots
set.seed(9370707) #Set random seed to have reproducible results
mat1_sample <- mat1[sample(62474, 1000),,drop=FALSE] #Keep only a sample of 1000 rows from the data

partial(xgb.model1, pred.var = "Age", ice = TRUE, 
              plot = TRUE, alpha = 0.1, plot.engine = "ggplot2",
              train = mat1_sample)

partial(xgb.model1, pred.var = "Age", ice = TRUE, center = TRUE,
              plot = TRUE, alpha = 0.1, plot.engine = "ggplot2", 
              train = mat1_sample)

partial(xgb.model1, pred.var = "Age", ice = TRUE, center = TRUE,
              plot = TRUE, alpha = 0.1, plot.engine = "ggplot2", 
              train = mat1_sample) + coord_cartesian(ylim = c(-2,1))

partial(xgb.model1, pred.var = "VehicleAge", ice = TRUE, 
              plot = TRUE, alpha = 0.1, plot.engine = "ggplot2", 
              train = mat1_sample)

partial(xgb.model1, pred.var = "VehicleAge", ice = TRUE, center = TRUE, 
              plot = TRUE, alpha = 0.1, plot.engine = "ggplot2", 
              train = mat1_sample)

partial(xgb.model1, pred.var = c("Age", "VehicleAge"), center = TRUE, 
              plot = TRUE, rug = TRUE, plot.engine = "ggplot2", chull = TRUE, 
              train = mat1_sample)

partial(xgb.model1, pred.var = c("MCClass", "VehicleAge"), center = TRUE, 
              plot = TRUE, rug = TRUE, plot.engine = "ggplot2", chull = TRUE, 
              train = mat1_sample)

partial(xgb.model1, pred.var = "MCClass", ice = TRUE, 
              plot = TRUE, alpha = 0.1, plot.engine = "ggplot2", 
              train = mat1_sample)


########################################################################
# 6. Loss Cost Modeling with a Tweedie distribution
# 6.1 Compute the average loss cost
avg_LC <- 
	with(dataOhlssonEN, sum(IncurredCost) / sum(EarnedExposure))

# 6.2 Gather all information in the XGBoost format
mat2.LossCost <- 	 
  xgb.DMatrix(
    mat1, #Matrix of explanatory variables
    label = dataOhlssonEN$IncurredCost / dataOhlssonEN$EarnedExposure, #Response Variable: the Loss Cost
	weight = dataOhlssonEN$EarnedExposure, #Weight Variable: Earned Exposure
	base_margin = rep(log(avg_LC), nrow(dataOhlssonEN)), #BaseRate
  )

# 6.3 Using Cross-Validation to find the optimal parameters
xgb.cv(
	data = mat2.LossCost,
	objective = 'reg:tweedie', #Like in a glm Tweedie, we'll model using a Tweedie distribution
	nrounds = 250, #Number of trees before stopping the algo
	max_depth = 3, #Size of the trees
	eta = 0.5, #Learning speed rate
	nfold = 5, #Train on 4/5 of the data and validate on the rest
	early_stopping_rounds = 5 #Computation will stop if the validation score doesn't improve in 5 rounds
  ) #For more info on parameters, visit https://xgboost.readthedocs.io/en/latest/parameter.html


########################################################################
# 7. Severity Modeling with a gamma distribution
# 7.1 Keep only data with claims for severity analysis
dataOhlssonENSev <- 
	dataOhlssonEN %>%
	filter(NbClaims > 0)

# 7.2 Matrix for Severity Data
mat1.Severity <- predict(dummy_model, newdata = dataOhlssonENSev)

# 7.3 Compute the average Severity that will be used as base rate
avg_Sev <- 
	with(dataOhlssonENSev, sum(IncurredCost) / sum(NbClaims))

# 7.4 Gather all information in the XGBoost format
mat2.Severity <- 	 
  xgb.DMatrix(
    mat1.Severity, #Matrix of explanatory variables
    label = dataOhlssonENSev$IncurredCost / dataOhlssonENSev$NbClaims, #Response Variable: the Severity
	weight = dataOhlssonENSev$NbClaims, #Weight Variable: Number of claims
	base_margin = rep(log(avg_Sev), nrow(dataOhlssonENSev)), #BaseRate
  )

# 7.5 Using Cross-Validation to find the optimal parameters
xgb.cv(
	data = mat2.Severity,
	objective = 'reg:gamma', #Like in a glm gamma, we'll model using a gamma distribution
	nrounds = 250, #Number of trees before stopping the algo
	max_depth = 3, #Size of the trees
	eta = 0.5, #Learning speed rate
	nfold = 5, #Train on 4/5 of the data and validate on the rest
	early_stopping_rounds = 5 #Computation will stop if the validation score doesn't improve in 5 rounds
  ) #For more info on parameters, visit https://xgboost.readthedocs.io/en/latest/parameter.html

