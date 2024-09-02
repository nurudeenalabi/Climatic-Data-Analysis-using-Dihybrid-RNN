#R codes on shrinkage methods for solar radiation model over Sokoto

setwd("C:/Users/NURUDEEN ALABI/Desktop")

solar=read.csv("solar.csv",header=T,na.strings="?");names(solar);attach(solar)
dim(solar);par(mfrow=c(1,1))
summary(solar);write.csv(summary(solar),"Solar_Radiation_Sokoto.csv")
fix(solar)

#Multiple Linear Regression

lm.fit=lm(sr~.,data=solar)
lm.fit
summary(lm.fit)
names(lm.fit)
coef(lm.fit)
confint(lm.fit)
vif(lm.fit)
#predict(lm.fit,data.frame(sr=(c(5,10,15))),interval="confidence")
#predict(lm.fit,data.frame(sr=(c(5,10,15))),interval="prediction")
plot(predict(lm.fit),residuals(lm.fit))
plot(predict(lm.fit),rstudent(lm.fit))
plot(hatvalues(lm.fit))
which.max(hatvalues(lm.fit))

# Ridge Regression

x=model.matrix(sr~.,solar)[,-1]
y=solar$sr
library(glmnet)
grid.sr=10^seq(10,-2,length=100)
ridge.sr=glmnet(x,y,alpha=0,lambda=grid.sr)
ridge.sr
dim(coef(ridge.sr))
set.seed(1)
train=sample(1:nrow(x),nrow(x)/2)
test=(-train)
y.test=y[test]
ridge.sr=glmnet(x[train,],y[train],alpha=0,
                lambda=grid,thresh=1e-12)#coefficient when lambda=ridge.solar$lambda which expected 
# to be smaller for large values of lambda
sqrt(sum(coef(ridge.sr)^2))
ridge.sr
set.seed(1)
cv.out.sr=cv.glmnet(x[train,],y[train],alpha=0)
cv.out.sr
plot(cv.out.sr)
bestlam.sr=cv.out.sr$lambda.min
bestlam.sr
ridge.pred.sr=predict(ridge.sr,s=bestlam.sr,newx=x[test,])
ridge.pred.sr
mean((ridge.pred.sr-y.test)^2)
out.sr=glmnet(x,y,alpha=0)
out.sr
predict(out.sr,type="coefficients",s=bestlam.sr)[1:7,]

# The Lasso

lasso.sr=glmnet(x[train,],y[train],alpha=1,lambda=grid.sr)
lasso.sr
sqrt(sum(abs(coef(lasso.sr))))
plot(lasso.sr)
set.seed(1)
cv.out.sr=cv.glmnet(x[train,],y[train],alpha=1)
cv.out.sr
plot(cv.out.sr)
bestlam.sr=cv.out.sr$lambda.min
bestlam.sr
lasso.pred.sr=predict(lasso.sr,s=bestlam.sr,newx=x[test,])
lasso.pred.sr
mean((lasso.pred.sr-y.test)^2)
out.sr=glmnet(x,y,alpha=1,lambda=grid.sr)
out.sr
lasso.coef.sr=predict(out.sr,type="coefficients",s=bestlam.sr)[1:7,]
lasso.coef.sr
lasso.coef.sr[lasso.coef.sr!=0]
lasso.coef.sr[lasso.coef.sr==0]

