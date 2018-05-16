library(gbm)
data(PimaIndiansDiabetes2,package='mlbench')

data <- PimaIndiansDiabetes2
library(VIM)

data_knn<-kNN(data)

summary(data_knn)
set.seed(2)
temp1<-data_knn[,1:9]
summary(data_knn)
dim(data_knn)
temp1<-data_knn[,1:9]
m=dim(temp1)
temp1[,9]<-as.factor(temp1[,9])

#number<-sample(m[1],500)
number<- sample(2, nrow(temp1), replace = TRUE, prob=c(0.8, 0.2))

train<-temp1[number==1,]
test<-temp1[number==2,]

train$diabetes<-as.numeric(train$diabetes)
train <- transform(train,diabetes=diabetes-1)
test$diabetes<-as.numeric(test$diabetes)
test<-transform(test,diabetes=diabetes-1)

train_1<-temp1[number==1,-1]
test_1<-temp1[number==2,-1]
train_3<-temp1[number==1,-3]
test_3<-temp1[number==2,-3]
train_7<-temp1[number==1,-7]
test_7<-temp1[number==2,-7]

train_1$diabetes<-as.numeric(train_1$diabetes)
test_1$diabetes<-as.numeric(test_1$diabetes)
train_1 <- transform(train_1,diabetes=diabetes-1)
test_1<-transform(test_1,diabetes=diabetes-1)

train_3$diabetes<-as.numeric(train_3$diabetes)
test_3$diabetes<-as.numeric(test_3$diabetes)
train_3 <- transform(train_3,diabetes=diabetes-1)
test_3<-transform(test_3,diabetes=diabetes-1)

train_7$diabetes<-as.numeric(train_7$diabetes)
test_7$diabetes<-as.numeric(test_7$diabetes)
train_7 <- transform(train_7,diabetes=diabetes-1)
test_7<-transform(test_7,diabetes=diabetes-1)
#-----------------------------------------------------
logit_fit<-glm(diabetes~.,data=train_7,family=binomial)
summary(logit_fit)
prob<-predict(logit_fit,type='response')
contrasts(train_7$diabetes)
pred<-rep('neg',602)
pred[prob>0.5]='pos'
table(pred,train_7$diabetes)
mean(pred==train_7$diabetes)



pred1<-predict(logit_fit,test_7,type='response')
bus=rep('neg',166)
bus[pred1>0.5]='pos'
table(bus,test_7$diabetes)
mean(bus==test_7$diabetes)

#---committe logitic---
m=dim(train)
n=500

lab_train<-matrix(0,1,m[1])
lab_test<-matrix(0,1,dim(test)[1])


for (i in c(1:n)){
imp<-sample(c(1:m[1]), 500, replace = TRUE)
d=train_7[imp,]  ########
logit_fit<-glm(diabetes~.,data=d,family=binomial)
prob<-predict(logit_fit,d,type='response')
pred<-rep(0,500)
pred[prob>0.5]=1

prob<-predict(logit_fit,train_7,type='response')
pred<-rep(-1,602)
pred[prob>0.5]=1

lab_train<-lab_train+(2*pred-1)

pred1<-predict(logit_fit,test_7,type='response')
bus=rep(0,166)
bus[pred1>0.5]=1
lab_test<-lab_test+(2*bus-1)
}


lab_train[which(lab_train>0)]=1
lab_train[-which(lab_train>0)]=0
lab_test[which(lab_test>0)]=1
lab_test[-which(lab_test>0)]=0

table(train$diabetes,lab_train)
table(test$diabetes,lab_test)
mean(lab_train==train$diabetes)
mean(lab_test==test$diabetes)
#-----logistic boost

model <- gbm(diabetes~.,data=train_7,shrinkage=0.01,
             distribution='bernoulli',cv.folds=5,
             n.trees=3000,verbose=F)
summary(model,order=FALSE)
# 用交叉检验确定最佳迭代次数
best.iter <- gbm.perf(model,method='cv')
best.iter
summary(model,best.iter)

plot.gbm(model,3,best.iter)

e_in<-predict(model,train_7,n.trees=best.iter)
e_in[which(e_in>0)]=1
e_in[-which(e_in>0)]=0
table(e_in,train[,9])
mean(e_in==train[,9])

pre<-predict(model,test_7,n.trees=best.iter)
pre[which(pre>0)]=1
pre[-which(pre>0)]=0
table(pre,test[,9])
mean(pre==test[,9])
#------------C5.0
train<-temp1[number==1,]
test<-temp1[number==2,]
train_1<-temp1[number==1,-1]
test_1<-temp1[number==2,-1]
train_3<-temp1[number==1,-3]
test_3<-temp1[number==2,-3]
train_7<-temp1[number==1,-7]
test_7<-temp1[number==2,-7]


library(C50)
model<-C5.0(train_7[,1:7],train[,9],trial=1,control = C5.0Control(CF=0.01))
model
summary(model)
	
plot(model)
pred<-predict(model,test_7,trial=1)
library(gmodels)
CrossTable(test[,9],pred,dnn=c('actual ','predict'))
	#----test error 0.1807229
table(pred,test[,9])
mean(pred==test[,9])

model<-C5.0(train[,1:8],train[,9],trial=1,
control = C5.0Control(CF=0.5))
pred<-predict(model,test)
mean(pred==test[,9])



n=100
err_in<-matrix(0,1,n)
err_out<-matrix(0,1,n)
for (i in c(1:n)){
model<-C5.0(train_7[,1:7],train[,9],trial=3,
control = C5.0Control(CF=i*0.01))

pred<-predict(model,test_7,trial=1)
err_out[i]<-mean(pred==test$diabetes)
err_in[i]<-1-model$boostResults[1,4]/100
}

plot(err_out[1,],type='b',main='C5.0 err',xlab='CF 0.01:0.01:1',
ylab='err',col=4,ylim=c(0.8,0.88))
par(new=T)
plot(err_in[1,],type='b',xlab='CF 0.01:0.01:1',ylab='err',
col=1,ylim=c(0.8,0.88))

leg.txt=c('black train_error','blue test_error')
legend(60, 0.25, leg.txt)

#--boost tree

num=30
err<-matrix(0,1,num)

for (i in c(1:num)){
model<-C5.0(train_7[,1:7],train[,9],trial=i
,control = C5.0Control(CF=.18))

pred<-predict(model,test_7)

err[i]<-mean(pred==test$diabetes)

}
plot(err[1,],type='b',xlab='boost time',ylab='accuracy',
main='C5.0 boost test accu,cf=0.45',ylim=c(0.75,0.9))
max(err)

pred<-predict(model,test_3)
table(pred,test$diabetes)
mean(pred==test$diabetes)

#####
model<-C5.0(train_7[,1:7],train[,9],trial=6,
control = C5.0Control(CF=0.18))
pred<-predict(model,test_7)
mean(pred==test$diabetes)
#---cart-----------
library(rpart)
library(rpart.plot)
library(rattle)


fit<-rpart(diabetes~.,data=train_7,control = rpart.control(cp = 0.005))
pred<-predict(fit,type = "class")
table(pred,train[,9])
mean(pred==train[,9])
plot(fit,main='Complexity parameter=0.02')
text(fit, use.n = TRUE)

pred1<-predict(fit,test_7,type='class')
table(pred1,test[,9])
mean(pred1==test[,9])

#0.03 0.02 81.89 86.75
#0.01
p.fit<-prune(fit,cp=0.03)


pred<-predict(p.fit,type = "class")
table(pred,train[,9])
mean(pred==train[,9])
#0.7956811

pred1<-predict(p.fit,test_1,type='class')
table(pred1,test[,9])
mean(pred1==test[,9])
#-------
library(e1071)

#cost  1  0.5 0.8253
#cost 0.4  0.8313
#cost  6.6 7.5 0.8373494
# cost 10 81927
#0.1 0.5 0.2 2
for (i in c(1:1000)){
fit<-svm(train_1[,1:7],train[,9],kernel='radial',cost=9.8+0.01*i,degree=3,epsilon=0.2)
prob<-predict(fit,train_1[,1:7])
mean(prob==train$diabetes)
pred<-predict(fit,test_1[,1:7])
table(pred,test$diabetes)
print(mean(pred==test$diabetes))
}


obj<-tune(svm,train[,1:8],train[,9],kernel='linear',
ranges=list(cost=c(1:20)*0.1,epsilon=c(1:20)*0.05))
#cost 1, epsilon 0.1
#0.8373494
pred<-predict(obj$best.model,test[,1:8])
table(pred,test[,9])
mean(pred==test[,9])

obj<-tune(svm,train[,1:8],train[,9],
ranges=list(cost=c(25:45)*0.1))
#cost0.5 epsilon 0.05

#0.8192771
#cost 0.9 epsilon 0.05
#0.8588


#聚类算法
#K-means  stats

#K-Medoids cluster

#系谱聚类（HC）stats

#密度聚类（DBSCAN）fpc

#期望最大化聚类（EM）mclust

library(stats)

? kmeans()
require(graphics)


(cl <- kmeans(train[,1:8], 3))

fitted.train <- fitted(cl);  head(fitted.train)
fit.test<-fitted()
resid.x <- x - fitted(cl)

cl$cluster

trainpart1<-train[cl$cluster==1,]

trainpart2<-train[cl$cluster==2,]
trainpart3<-train[cl$cluster==3,]


cl$centers
(cltest <- kmeans(test[,1:8], cl$centers,iter.max = 1))
test1<-test[cltest$cluster==1,]
test2<-test[cltest$cluster==2,]
test3<-test[cltest$cluster==3,]

temp<-rbind(trainpart1,trainpart3)
temptest<-rbind(test1,test3)

(clvalid <- kmeans(valid[,1:8], cl$centers,iter.max = 1))
valid1<-valid[clvalid$cluster==1,]
valid2<-valid[clvalid$cluster==2,]
valid3<-valid[clvalid$cluster==3,]


#--------------
library(rpart)
library(rpart.plot)
library(rattle)
temp$diabetes=as.factor(temp$diabetes)
temptest$diabetes=as.factor(temptest$diabetes)


fit<-rpart(diabetes~.,data=temp,control = rpart.control(cp = 0.01))
plot(fit,main='Complexity parameter=0.03')
text(fit, use.n = TRUE)

fancyRpartPlot(fit,main='part1&3 Complexity parameter=0.01')

pred<-predict(fit,type = "class")
table(pred,temp[,9])
mean(pred==temp[,9])


pred1<-predict(fit,temptest,type='class')
table(pred1,temptest[,9])
print(mean(pred1==temptest[,9]))

pred<-predict(fit,valid2,type = "class")
table(pred,valid2[,9])
mean(pred==valid2[,9])



#0.0 0.01 12
#----
# 5 part2 
fit<-rpart(diabetes~.,data=trainpart2,control = rpart.control(cp = 0.01))
plot(fit,main='Complexity parameter=0')
text(fit, use.n = TRUE)

pred<-predict(fit,type = "class")
table(pred,trainpart2[,9])
mean(pred==trainpart2[,9])


pred1<-predict(fit,test2,type='class')
table(pred1,test2[,9])
print(mean(pred1==test2[,9]))
#------------------------

#svm no
library(e1071)
for (i in c(1:100)){
fit<-svm(temp[,1:8],temp[,9],kernel='radial',type='C',
cost=0.01+0.1*i,degree=4,epsilon=0.01)
prob<-predict(fit,temp[,1:8])
mean(prob==temp$diabetes)
pred<-predict(fit,temptest[,1:8])
table(pred,temptest$diabetes)
print(mean(pred==temptest$diabetes))
}


##
temp$diabetes<-as.numeric(temp$diabetes)
temp <- transform(temp,diabetes=diabetes-1)
temptest$diabetes<-as.numeric(temptest$diabetes)
temptest<-transform(temptest,diabetes=diabetes-1)

model <- gbm(diabetes~.,data=temp,shrinkage=0.01,
             distribution='bernoulli',cv.folds=5,
             n.trees=3000,verbose=F)
summary(model,order=FALSE)
# 用交叉检验确定最佳迭代次数
best.iter <- gbm.perf(model,method='cv')
best.iter

plot.gbm(model,3,best.iter)

e_in<-predict(model,temp,n.trees=best.iter)
e_in[which(e_in>0)]=1
e_in[-which(e_in>0)]=0
table(e_in,temp[,9])

best.iter
for (i in c(1:700)){
pre<-predict(model,temptest,n.trees=200)
pre[which(pre>0)]=1
pre[-which(pre>0)]=0

table(pre,temptest[,9])
print(mean(pre==temptest[,9]))
}
##
#--------------------
library(C50)
for (i in c(1:95)){
model<-C5.0(temp[,1:8],temp[,9],trial=17,
control = C5.0Control(CF=0.5))
model
summary(model)
	#---train error 12.6%
#for (i in c(1:17)){

pred<-predict(model,temptest,trial=i)

table(pred,temptest[,9])
print(mean(pred==temptest[,9]))

}

#8 4 0.5 
#87 81 93.1%   cf 0.01  tran#15.6% 6
#17
#----------------------------

for (i in c(1:17)){
model<-C5.0(trainpart2[,1:8],trainpart2[,9],trial=17,
control = C5.0Control(CF=0.01))
model
summary(model)
	#---train error 12.6%
for (i in c(1:17)){

pred<-predict(model,test2,trial=i)

table(pred,test2[,9])
print(mean(pred==test2[,9]))

}

#----
for (i in c(1:17)){
model<-C5.0(trainpart3[,1:8],trainpart3[,9],trial=1,
control = C5.0Control(CF=0.3))
model
summary(model)
	#---train error 12.6%
for (i in c(1:17)){

pred<-predict(model,test3,trial=i)

table(pred,test3[,9])
print(mean(pred==test3[,9]))

}




