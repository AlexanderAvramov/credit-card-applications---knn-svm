#################################################################################################################
# description: to train a model predicting credit card application approvals using anonymized data
#              the models used are knn with cross validation & svm with cross validation
# data source: https://archive.ics.uci.edu/ml/datasets/Credit+Approval
#################################################################################################################

########################################################################
# SETUP
########################################################################
setwd("..")

rm(list = ls())
library(dplyr)
library(tidyr)
library(kernlab)
library(kknn)
library(mltools)
library(data.table)
library(caret)

options(scipen = 99999)

########################################################################
# READ IN THE DATA
########################################################################
df_r <- read.delim("input/crx.data.txt", sep = ",",stringsAsFactors = FALSE, header = FALSE)


########################################################################
# DATA PROCESSING PRIOR TO SPLITTING DATA
########################################################################
# convert numerics to numerics and standardize response variable
df <- df_r %>%
  mutate(R1 = ifelse(V16 %in% c("+"), 1,0)) %>%
  select(-V16)


# turn ?s to NAs
nmz <- names(df)
for (i in 1:length(nmz)){
  df[,nmz[i]] <- gsub("\\?",NA,df[,nmz[i]])
}


# get rid of observations where any variable is missing
df$missing <- rowSums(is.na(df))
sum(df$missing)

df <- df %>%
  filter(missing %in% c(0)) %>%
  select(-missing)

stopifnot(sum(is.na(df))==0)


########################################################################
# SPLIT INTO TRAIN AND TEST
########################################################################
set.seed(22)
nrows <- dim(df)[1]
smp <- sample(nrows,nrows*.85)
df_train <- df[smp,]
df_test <- df[-smp,]


########################################################################
# DATA PROCESSING AFTER TO SPLITTING DATA (AFTER TO AVOID DATA LEAKAGE)
########################################################################

#############################
# TRAIN DATA
#############################
df_train_res <- df_train %>%
  mutate(id = 1:n(),
         R1 = as.factor(R1)) %>%
  select(id,R1)

df_train_num <- df_train %>%
  mutate(id = 1:n()) %>%
  select(id,V2,V3,V8,V11,V14,V15)

df_train_num <- data.frame(lapply(df_train_num, function(x) as.numeric(x)))

df_train_cat <- df_train %>%
  mutate(id = 1:n()) %>%
  select(id,V1,V4,V5,V6,V7,V9,V10,V12,V13) %>%
  mutate(V1 = as.factor(V1),
         V4 = as.factor(V4),
         V5 = as.factor(V5),
         V6 = as.factor(V6),
         V7 = as.factor(V7),
         V9 = as.factor(V9),
         V10 = as.factor(V10),
         V12 = as.factor(V12),
         V13 = as.factor(V13))

df_train_cat <- as.data.frame(one_hot(as.data.table(df_train_cat)))
df_train_cat <- df_train_cat[,-nearZeroVar(df_train_cat)]

df_train_final <- df_train_res %>%
  full_join(df_train_num, by = "id") %>%
  ungroup() %>%
  full_join(df_train_cat, by = "id") %>%
  ungroup() %>%
  select(-id)

stopifnot(dim(df_train_final)[1] == dim(df_train)[1])

rm(list = ls()[!ls() %in% c("df_train_final", "df_test", "df")])


#############################
# TEST DATA
#############################
df_test_res <- df_test %>%
  mutate(id = 1:n(),
         R1 = as.factor(R1)) %>%
  select(id,R1)

df_test_num <- df_test %>%
  mutate(id = 1:n()) %>%
  select(id,V2,V3,V8,V11,V14,V15)

df_test_num <- data.frame(lapply(df_test_num, function(x) as.numeric(x)))

df_test_cat <- df_test %>%
  mutate(id = 1:n()) %>%
  select(id,V1,V4,V5,V6,V7,V9,V10,V12,V13) %>%
  mutate(V1 = as.factor(V1),
         V4 = as.factor(V4),
         V5 = as.factor(V5),
         V6 = as.factor(V6),
         V7 = as.factor(V7),
         V9 = as.factor(V9),
         V10 = as.factor(V10),
         V12 = as.factor(V12),
         V13 = as.factor(V13))

df_test_cat <- as.data.frame(one_hot(as.data.table(df_test_cat)))
df_test_cat <- df_test_cat[,-nearZeroVar(df_test_cat)]

df_test_final <- df_test_res %>%
  full_join(df_test_num, by = "id") %>%
  ungroup() %>%
  full_join(df_test_cat, by = "id") %>%
  ungroup() %>%
  select(-id)

stopifnot(dim(df_test_final)[1] == dim(df_test)[1])

rm(list = ls()[!ls() %in% c("df_train_final", "df_test_final")])


#############################
# STANDARDIZE VARIABLES
#############################
features <- intersect(names(df_train_final),names(df_test_final))

df_test_final <- df_test_final[,features]
df_train_final <- df_train_final[,features]


########################################################################
# PART 1: KNN MODEL WITH CROSS VALIDATION USING CARET
########################################################################
# train the model using the train data and cross validation of k = 5
knn_model <- train(R1~.,
               method = "knn",
               tuneGrid = expand.grid(k = 1:100),
               trControl = trainControl(method = "cv", number = 5),
               metric = "Accuracy",
               data = df_train_final,
               preProc = c("center","scale"))

knn_model

# test the model on the test set
predict <- predict(knn_model, df_test_final)
accuracy <- sum(predict == df_test_final[,c("R1")]) / dim(df_test_final)[1]
accuracy
confusionMatrix(predict, df_test_final[,c("R1")])


####################################################
# PART 2: SVM MODEL USING CARET
####################################################
svm_model <- train(R1~.,
                   method = "svmLinear",
                   tuneGrid = expand.grid(C = seq(1, 2.5, length = 100)),
                   trControl = trainControl(method = "cv", number = 5),
                   metric = "Accuracy",
                   data = df_train_final,
                   preProc = c("center","scale"))

svm_model

# the tuning parameter C imposes a penalty for missclasifying: 
# the higher the value of C, the less likely it missclasifies in training
# the high value of C in the loss function of the model with the highest accuracy signifies
# that we place a high importance on correctly classifying the points as opossed to increasing the margin

# test the model on the test set
predict <- predict(svm_model, df_test_final)
accuracy <- sum(predict == df_test_final[,c("R1")]) / dim(df_test_final)[1]
accuracy
confusionMatrix(predict, df_test_final[,c("R1")])