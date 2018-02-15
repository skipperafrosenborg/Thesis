#install.packages("glmnet")
library(glmnet)

library(readr)
### CPU data set ###
machine <- read_csv("~/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/Data/machine.data", 
                    col_names = FALSE)
#View(machine)
machine = machine[,3:9]

x = machine[,1:6]
y= machine[,7]

x = data.matrix(x)
y = data.matrix(y)

x_train = x[1:104,]
x_test = x[105:209,]

y_train = y[1:104,]
y_test = y[105:209,]

fit.lasso <- glmnet(x_train,y_train,intercept=TRUE)

pred = predict(fit.lasso, x_test)

rSquared <- 1-apply((y_test-pred)^2, 2, sum)/sum((y_test-mean(y_test))^2)
plot(log(fit.lasso$lambda), rSquared, type="b", xlab="Log(lambda)")

fit.lasso$lambda[38]

lam.best <- fit.lasso$lambda[which.max(rSquared)]
lam.best
beta = coef(fit.lasso, s=lam.best)
beta
max(rSquared)

### Elevator data set ###
elevator_X <- read_csv("~/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/Data/Elevators/elevatorXTrain.CSV", 
                    col_names = FALSE)
elevator_X[is.na(elevator_X)] = 0
elevator_test_X <- read_csv("~/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/Data/Elevators/elevatorXTest.CSV", 
                          col_names = FALSE)
elevator_test_X[is.na(elevator_test_X)] = 0
elevator_Y <- read_csv("~/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/Data/Elevators/elevatorYTrain.CSV", 
                       col_names = FALSE)
elevator_Y[is.na(elevator_Y)] = 0
elevator_test_Y <- read_csv("~/Google Drive/DTU/10. Semester/Thesis/GitHubCode/Thesis/Data/Elevators/elevatorYTest.CSV", 
                            col_names = FALSE)
elevator_test_Y[is.na(elevator_test_Y)] = 0

x = elevator_X
y = elevator_Y
x = data.matrix(x)
y = data.matrix(y)
x_train = x
y_train = y[ ,1]

x = elevator_test_X
y = elevator_test_Y
x = data.matrix(x)
y = data.matrix(y)
x_test = x
y_test = y[ ,1]

fit.lasso <- glmnet(x_train,y_train,intercept=TRUE)

pred = predict(fit.lasso, x_test)

rSquared <- 1-apply((y_test-pred)^2, 2, sum)/sum((y_test-mean(y_test))^2)
plot(log(fit.lasso$lambda), rSquared, type="b", xlab="Log(lambda)")

fit.lasso$lambda[which.max(rSquared)]

lam.best <- fit.lasso$lambda[which.max(rSquared)]
lam.best
beta = coef(fit.lasso, s=lam.best)
beta
max(rSquared)