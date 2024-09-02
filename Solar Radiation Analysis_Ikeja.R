#Analysis of Solar Radiation Clearness Index Models at Ikeja between 1980 to 2010
#By Alabi Nurudeen*(Dept.of Statistics, YABATECH),Ilori Babatunde Abdulmalik**(Dept of Physics, YABATECh)
#email:nurudene.alabi@gmail.com*;mobile:08152522516*,iloribabatunde7@gmail.com**,mobile:08033543116
#Methods1: Supervised Learning;Quadratic,Cubic & Quartic linear Models,
#Methods2: Supervised Learning:Generalized Additive Models;Splines-Natural Splines & Smoothing Splines
#Copyright 2016.



#we invoked required packages from various R libraries
require(iterators);require(foreach);require(splines);library(gam);library(ISLR)
library(MASS);library(lattice);require(akima);require(boot)

src=read.csv("src.csv",header=TRUE);attach(src);src;dim(src);names(src);attach(src);summary(src);par(mfrow=c(1,1));
plot(kt,col="magenta3",type="l",lwd=1,pch=18,main="Clearness Index (Solar Radiation) at Ikeja (Period:1980-2010")
plot(year,kt,col=(y+2),pch=18,main="Clearness index (Solar Radiation) at Ikeja (Period:1980-2010")
plot(ks,col="red",type="l",lwd=1,pch=18,main="Clearness index (Sunshine Hours) at Ikeja (Period:1980-2010")
plot(year,ks,col=(y+2),pch=18,main="Clearness Index (Sunshine Hours) at Ikeja (Period:1980-2010")
plot(rh,col="maroon4",type="l",lwd=1,main="Relative Humidity at Ikeja")
plot(year,rh,col=(y+2),pch=18,main="Relative Humidity at Ikeja Period:1980-2010")
plot(year,te,col=(y+4),pch=18,main="Temprature at Ikeja Period:1980-2010")
plot(te,type="l",col="navy",lwd=2,pch=18,main="Temprature at Ikeja (Period:1980-2010")
plot(year,rh,col=(y+5),pch=18,main="Relative Humidity at Ikeja Period:1980-2010")
plot(rh,type="l",col="tan1",lwd=2,pch=18,main="Relative Humidity at Ikeja (Period:1980-2010")
plot(ep,type="l",col="violet",lwd=1,pch=18,main="Evaporation Piche at Ikeja Period:1980-2010")
plot(year,ep,col=(y+9),pch=18,main="Evaporation Piche at Ikeja (Period:1980-2010")
plot(year,ws,col=(y+6),pch=18,main="Wind Speed at Ikeja Period:1980-2010")
plot(ws,type="l",col="aquamarine",lwd=2,pch=18,main="Wind Speed at Ikeja (Period:1980-2010")
plot(ks,kt,col=(y+2),pch=20,main="Scatter Plot of Sunshine Hours and Solar Radiation at Ikeja")
plot(te,kt,col=(y+2),pch=20,main="Scatter Plot of Temperature and Solar Radiation at Ikeja")
plot(rh,kt,col=(y+2),pch=20,main="Scatter Plot of Relative Humidity and Solar Radiation at Ikeja")
plot(ep,kt,col=(y+2),pch=20,main="Scatter Plot of Evaporation Piche and Solar Radiation at Ikeja")
plot(ws,kt,col=(y+2),pch=20,main="Scatter Plot of Wind Speed and Solar Radiation at Ikeja")

#Linear Regression on Solar Radiation on clearness index and temperature at Ikeja, Lagos
lm.kt1=lm(kt~ks+te);summary(lm.kt1)$coef;summary(lm.kt1);names(lm.kt1);coef(lm.kt1);par(mfrow=c(2,2))
confint(lm.kt1)
predict(lm.kt1,data.frame(ks,te), interval="confidence")
predict(lm.kt1,data.frame(ks,te), interval="prediction")
plot(ks,kt,col=(y+3),pch=20)
plot(te,kt,col=(y+3),pch=20)
plot(ks,kt,col="red");plot(ks,kt,col=(y+3),pch=20)
plot(te,kt,col="blue",pch="+");par(mfrow=c(2,2))
plot(lm.kt1);lines(predict(lm.kt1),col="red",pch=20)
plot(predict(lm.kt1),residuals(lm.kt1,col=(y+4)),pch=20)
plot(predict(lm.kt1),rstudent(lm.kt1),pch=20,col=(y+3))
plot(hatvalues(lm.kt1))
which.max(hatvalues(lm.kt1))

#Linear Regression on Solar Radiation against clearness index,temperature,relative humidity,evaporation piche,wind speed at Ikeja, Lagos
lm.kt2=lm(kt~ks+te+rh+ep+ws);summary(lm.kt2);names(lm.kt2);coef(lm.kt2)
confint(lm.kt2)
predict(lm.kt2,data.frame(ks,te,rh,ep,ws), interval="confidence")
predict(lm.kt2,data.frame(ks,te,rh,ep,ws), interval="prediction")
plot(rh,kt,col=(y+3),pch=20);plot(ep,kt,col=(y+3),pch=20)
plot(ws,kt,col=(y+3),pch=20);plot(rh,kt,col="red")
plot(rh,kt,col=(y+3),pch=20);plot(ep,kt,col=(y+3),pch="+")
par(mfrow=c(2,2));plot(lm.kt2,col=(y+3));lines(predict(lm.kt2),col="red",pch=20)
plot(predict(lm.kt2),residuals(lm.kt2),col=(y+3),pch=20)
 plot(predict(lm.kt2),rstudent(lm.kt2),pch=20,col=(y+3))
plot(hatvalues(lm.kt2));which.max(hatvalues(lm.kt2))

#Quadratic Polynomial Regression on Solar Radiation at Ikeja, Lagos
lm.kt3=lm(kt~poly(ks,2)+poly(te,2),data=(src));summary(lm.kt3)
names(lm.kt3);coef(lm.kt3)
lm.kt4=lm(kt~poly(ks,2)+poly(te,2)+rh+poly(ep,2)+ws,data=(src));summary(lm.kt4)
names(lm.kt4);coef(lm.kt4)
confint(lm.kt4)
pred.quad1=predict(lm.kt4,data.frame(ks,te,rh,ep,ws), interval="confidence")
pred.quad2=predict(lm.kt4,data.frame(ks,te,rh,ep,ws), interval="prediction")
plot(lm.kt4,col=(y+3));lines(predict(lm.kt4),col="red",pch=20)
pred.quad1;pred.quad2
plot(predict(lm.kt4),residuals(lm.kt4),col=(y+3),pch=20)
plot(predict(lm.kt4),rstudent(lm.kt4),pch=20,col=(y+3))

#Cubic Polynomial Regression on Solar Radiation at Ikeja, Lagos
lm.kt5=lm(kt~poly(ks,3)+poly(te,3),data=(src));summary(lm.kt5)
names(lm.kt5);coef(lm.kt5)
lm.kt6=lm(kt~poly(ks,3)+poly(te,3)+poly(rh,3)+poly(ep,3)+poly(ws,3),data=(src))
summary(lm.kt6);names(lm.kt6);coef(lm.kt6)
confint(lm.kt6)
pred.cubic1=predict(lm.kt6,data.frame(ks,te,rh,ep,ws), interval="confidence")
pred.cubic2=predict(lm.kt6,data.frame(ks,te,rh,ep,ws), interval="prediction")
plot(lm.kt6,col=(y+3));lines(predict(lm.kt6),col="red",pch=20)
pred.cubic1
pred.cubic2
plot(predict(lm.kt6),residuals(lm.kt6),col=(y+3),pch=20)
plot(predict(lm.kt6),rstudent(lm.kt6),pch=20,col=(y+3))

#Quartic Polynomial Regression on Solar Radiation at Ikeja, Lagos
lm.kt7=lm(kt~poly(ks,4)+poly(te,4),data=(src));summary(lm.kt7);names(lm.kt7);coef(lm.kt7)
lm.kt8=lm(kt~poly(ks,4)+poly(te,4)+poly(rh,4)+poly(ep,4)+poly(ws,4),data=(src))
summary(lm.kt8)
names(lm.kt8);coef(lm.kt8)
confint(lm.kt8)
pred.quartic1=predict(lm.kt8,data.frame(ks,te,rh,ep,ws),interval="confidence")
pred.quartic2=predict(lm.kt8,data.frame(ks,te,rh,ep,ws),interval="prediction")
plot(lm.kt8,col=(y+3));lines(predict(lm.kt8),col="red",pch=20)
pred.quartic1
pred.quartic2
plot(predict(lm.kt8),residuals(lm.kt8),col=(y+3),pch=20)
plot(predict(lm.kt8),rstudent(lm.kt8),pch=20,col=(y+3))

#Quindratic Polynomial Regression on Solar Radiation at Ikeja, Lagos
lm.kt9=lm(kt~poly(ks,5)+poly(te,5),data=(src));summary(lm.kt9);names(lm.kt9);coef(lm.kt9)
lm.kt10=lm(kt~poly(ks,5)+poly(te,5)+poly(rh,5)+poly(ep,5)+poly(ws,5),data=(src));summary(lm.kt10)
names(lm.kt10);coef(lm.kt10)
confint(lm.kt10)
pred.quin1=predict(lm.kt10,data.frame(ks,te,rh,ep,ws),interval="confidence")
pred.quin2=predict(lm.kt10,data.frame(ks,te,rh,ep,ws), interval="prediction")
plot(lm.kt10,col=(y+3));lines(predict(lm.kt10),col="red",pch=20)
pred.quin1;pred.quin2
plot(predict(lm.kt10),residuals(lm.kt10),col=(y+3),pch=20)
plot(predict(lm.kt10),rstudent(lm.kt10),pch=20,col=(y+3))

#Analysis of Variance amongst the models
anova(lm.kt1,lm.kt3,lm.kt5,lm.kt7,lm.kt9)
anova(lm.kt2,lm.kt4,lm.kt6,lm.kt8,lm.kt10)
anova(lm.kt1,lm.kt3,lm.kt5,lm.kt7,lm.kt9,lm.kt2,lm.kt4,lm.kt6,lm.kt8,lm.kt10)

#Generalized Additive Model on Solar Radiation Clearness Index data


library(ISLR);library(MASS);library(splines);library(locfit)

#we fit a generalized additive model GAM to predict
#Solar Radiation using the natural spline functions of the 
#year,sunshine index,relative humidity,.......
#since this is a big linear regression model using an appropriate 
#choice of basis functions, we simply used the lm() function as follows:

attach(src);gam.kt1=lm(kt~ns(ks,4)+ns(te,4)+ns(rh,4)+ns(ep,4)+ns(ws,4))
summary(gam.kt1)


#we now fit the model introducing the smoothing splines instead of the natural 
#spline. Smoothing splines allow more general sorts of GAMs to fitted. we invoked
# the gam library. s() function which is part of the gam library is used to indicate 
#that smoothing spline is required.

lm.kt10=lm(kt~poly(ks,5)+poly(te,5)+poly(rh,5)+poly(ep,5)+poly(ws,5),data=(src))
summary(lm.kt10);names(lm.kt10);coef(lm.kt10)
gam.kt2=gam(kt~s(ks,2)+s(te,2)+s(rh,2)+s(ep,2)+s(ws,2))
summary(gam.kt2);par(mfrow=c(1,2))
plot(gam.kt2,se=TRUE,col="blue",lwd=3)


#the generic plot() function recognizes that gam2 is an object of class gam, and invokes
#the appropriate plot.gam() method. Even though gam.kt1 is not of class gam but rather class lm
#we can still use plot.gam() on it.
gam.kt1=lm(kt~ns(ks,2)+ns(te,2)+ns(rh,2)+ns(ep,2)+ns(ws,2))
plot.gam(gam.kt1,se=T,col="red",lwd=3)
preds.ns=predict(gam.kt2,newdata=src);preds.ns

pred.quin1=predict(lm.kt10,data.frame(ks,te,rh,ep,ws), interval="confidence")
pred.quin2=predict(lm.kt10,data.frame(ks,te,rh,ep,ws), interval="prediction")
summary(lm.kt10);summary(gam.kt2)
preds.s=predict(gam.kt2,newdata=src)
preds.s
pred.quin1
pred.quin2
par(mfrow=c(2,2))
plot(kt,col=(y+3),pch="+",main="Clearness Index (Actual Versus Predicted)-GAM (Smonthing Spline)")
lines(kt,col="green",lwd=2)
lines(preds.s,col="orange",lwd=1)
legend("topleft",legend=c("Actual","Predicted"),col=c("blue","red"),lty=1,cex=0.8)

#local regression for the ....... term with a span

gam.lo=gam(kt~lo(ks,span=0.8)+lo(te,span=0.8)+lo(rh,span=0.8)+lo(ep,span=0.8)+lo(ws,span=0.8),data=src)
plot.gam(gam.lo,se=T,col="blue",lwd=3)
pred.lo=predict(gam.lo,newdata=src);pred.lo
par(mfrow=c(2,2))
plot(kt,col=(y+3),pch=20,main="Clearness Index (Actual Versus Predicted)-Local Regression")
lines(pred.lo,col="red",lwd=3)
lines(kt,col="blue",lwd=2)
legend("topleft",legend=c("Actual","Predicted"),col=c("blue","red"),lty=1,cex=0.8)

#lo() function can also be used to create interactions before calling the gam()function
#which fits a two-term model in which the first term is an interaction between ks and te
# fit by a local rgression surface. this can be plotted using the akima package

gam.lo.i1=gam(kt~lo(ks,te,span=0.2),data=src);summary(gam.lo.i1)
gam.lo.i2=gam(kt~lo(ks,rh,span=0.2),data=src);summary(gam.lo.i2)
gam.lo.i3=gam(kt~lo(ks,ep,span=0.2),data=src);summary(gam.lo.i3)
gam.lo.i4=gam(kt~lo(te,rh,span=0.2),data=src);summary(gam.lo.i4)
gam.lo.i5=gam(kt~lo(te,ws,span=0.2),data=src);summary(gam.lo.i5)
gam.lo.i6=gam(kt~lo(rh,ep,span=0.2),data=src);summary(gam.lo.i6)
gam.lo.i7=gam(kt~lo(rh,ws,span=0.2),data=src);summary(gam.lo.i7)
gam.lo.i8=gam(kt~lo(ep,ws,span=0.2),data=src);summary(gam.lo.i8)

library(akima)
plot(gam.lo.i1)
plot(gam.lo.i2)
plot(gam.lo.i3)
plot(gam.lo.i4)
plot(gam.lo.i5)
plot(gam.lo.i6)
plot(gam.lo.i7)
plot(gam.lo.i8)


#we divided the data to train the selected models i.e quindratic and gam.kt2 
#to test the performances of the two models

#lm.kt10=lm(kt~poly(ks,5)+poly(te,5)+poly(rh,5)+poly(ep,5)+poly(ws,5),data=(src)).......polynomial model-Quindratic

#gam.kt2=gam(kt~s(ks,2)+s(te,2)+s(rh,2)+s(ep,2)+s(ws,2),data=test.2005)..........GAM-Smoothing Spline


train=(year<2005)
train
test.2005=src[!train,]
test.2005
dim(test.2005)
kt.2005=kt[!train]
dim(kt.2005)

#the quindratic is favoured for this data set due to lower residual standard error and improved cofficient of 
#multiple determination
lm.kt10=lm(kt~poly(ks,5)+poly(te,5)+poly(rh,5)+poly(ep,5)+poly(ws,5),subset=c(year<2005))
summary(lm.kt10);names(lm.kt10);coef(lm.kt10)
gam.kt2=gam(kt~s(ks,2)+s(te,2)+s(rh,2)+s(ep,2)+s(ws,2),subset=train)
summary(gam.kt2);par(mfrow=c(2,2))
plot(gam.kt2,se=TRUE,col="red",lwd=3,subset=c(year<2005))
confint(lm.kt10)
pred.quin1=predict(lm.kt10,test.2005,type="response",interval="confidence")
pred.quin2=predict(lm.kt10,test.2005, type="response",interval="prediction")
plot(lm.kt10,col=(y+3));lines(predict(lm.kt10),col="red",pch=20)
pred.quin1
pred.quin2
plot(predict(lm.kt10),residuals(lm.kt10),col=(y+3),pch=20)
plot(predict(lm.kt10),rstudent(lm.kt10),pch=20,col=(y+3))

predict(gam.kt2,test.2005,type="response",interval="confidence")
predict(gam.kt2,test.2005,type="response",interval="prediction")
write.csv(predict(gam.kt2,test.2005,type="response"),"gamtest.csv")
write.csv(pred.quin2,"quintest.csv")
#End!!!!!

