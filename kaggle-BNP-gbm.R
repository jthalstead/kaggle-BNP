##### Best Score (logloss): 0.47303
##### RF Benchmark: 0.50602 | 0.49214
##### All Preds = 0.5: 0.69315

rm(list=ls())
library(gbm)
library(caret)

na.roughfix2 = function (object, ...) {
  res = lapply(object, roughfix)
  structure(res, class = "data.frame", row.names = seq_len(nrow(object)))
}

roughfix = function(x) {
  missing = is.na(x)
  if (!any(missing)) return(x)
  
  if (is.numeric(x)) {
    x[missing] = median.default(x[!missing])
  } else if (is.factor(x)) {
    freq = table(x)
    x[missing] = names(freq)[which.max(freq)]
  } else {
    stop("na.roughfix only works for numeric or factor")
  }
  x
}

ds = read.csv(file.choose())
uniq = sapply(ds, function(x) length(unique(x)))

### target
ds.y = ds[,colnames(ds) == "target"]

### factors
ds.fact = ds[, (uniq<20000) & colnames(ds) != "target"]
ds.fact = sapply(ds.fact, function(x) as.factor(x))
ds.fact = as.data.frame(ds.fact)
# v22: Factor w/ 13,272 levels
ds.fact$v22a = as.factor(substr(ds.fact$v22,1,1))
ds.fact$v22b = as.factor(substr(ds.fact$v22,2,2))
ds.fact$v22c = as.factor(substr(ds.fact$v22,3,3))
ds.fact$v22d = as.factor(substr(ds.fact$v22,4,4))
# v56: Factor w/ 106 levels
ds.fact$v56a = as.factor(substr(ds.fact$v56,1,1))
ds.fact$v56b = as.factor(substr(ds.fact$v56,2,2))
# v125: Factor w/ 90 levels
ds.fact$v125a = as.factor(substr(ds.fact$v125,1,1))
ds.fact$v125b = as.factor(substr(ds.fact$v125,2,2))

### numeric
ds.int = ds[, (uniq>20000) & colnames(ds) != 'ID']
ds.int = round(log(ds.int+1), 8)

### combine
ds.all = cbind(ds.y, ds.fact, ds.int)
ds.all = ds.all[, !(colnames(ds.all) %in% c('v22','v56','v125'))]
ds.samp = ds.all[(sample(1:nrow(ds.all), (nrow(ds.all)*0.25))), ]

# library(corrgram)
# corrgram(ds.samp, order=TRUE)

bnp.gbm = gbm(ds.y ~ ., 
              data = ds.samp,
              distribution = 'bernoulli', 
              n.trees = 5000, 
              interaction.depth = 8,
              shrinkage = 0.001,
              train.fraction = 0.6,
              bag.fraction = 0.5,
              verbose = T)

summary(bnp.gbm)
gbm.perf(bnp.gbm)

# var.worth = cbind(summary(bnp.gbm)[1],summary(bnp.gbm)[2]>0.3)
# var.worth = var.worth[var.worth$rel.inf==1,1]
# ds.worth = ds.samp[, (colnames(ds.samp) %in% var.worth) | (colnames(ds.samp)=='ds.y')]

ds.samp$preds = round(predict(bnp.gbm, newdata = ds.samp, type = 'response'))
confusionMatrix(ds.samp$preds, ds.samp$ds.y)

ds.test = read.csv(file.choose())
uniq.test = sapply(ds.test, function(x) length(unique(x)))

### factors
ds.test.fact = ds.test[, (uniq.test<20000)]
ds.test.fact = sapply(ds.test.fact, function(x) as.factor(x))
ds.test.fact = as.data.frame(ds.test.fact)
# v22: Factor w/ 13,272 levels
ds.test.fact$v22a = as.factor(substr(ds.test.fact$v22,1,1))
ds.test.fact$v22b = as.factor(substr(ds.test.fact$v22,2,2))
ds.test.fact$v22c = as.factor(substr(ds.test.fact$v22,3,3))
ds.test.fact$v22d = as.factor(substr(ds.test.fact$v22,4,4))
# v56: Factor w/ 106 levels
ds.test.fact$v56a = as.factor(substr(ds.test.fact$v56,1,1))
ds.test.fact$v56b = as.factor(substr(ds.test.fact$v56,2,2))
# v125: Factor w/ 90 levels
ds.test.fact$v125a = as.factor(substr(ds.test.fact$v125,1,1))
ds.test.fact$v125b = as.factor(substr(ds.test.fact$v125,2,2))

### numeric
ds.test.int = ds.test[, (uniq.test>20000) & colnames(ds.test) != 'ID']
ds.test.int = round(log(ds.test.int+1), 4)

### combine
ds.test.all = cbind(ds.test.fact, ds.test.int)
ds.test.all = ds.test.all[, !(colnames(ds.test.all) %in% c('v22','v56','v125'))]
ds.test.all$PredictedProb = predict(bnp.gbm, newdata = ds.test.all, type = 'response')
ds.test.all$ID = ds.test$ID
ds.submit = cbind(ds.test.all$ID, ds.test.all$PredictedProb)
ds.submit = as.data.frame(ds.submit)
colnames(ds.submit) = c("ID","PredictedProb")
write.csv(ds.submit, file = file.choose(), row.names = FALSE)

