function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
[m,s] = size(X); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
sum=0;
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
for i=1:m
a=X(i,:);
a=a';
e=theta'*a;
prediction=sigmoid(e);
p=-y(i)*log(prediction);
n=-(1-y(i))*log(1-prediction);
J=J+p+n;
end
J=J/m;

for j=2:s
    sum=sum+(theta(j)^2);
end
sum=sum*(lambda/(2*m));
J=J+sum;


for j=2:s
sum=0;
for i=1:m
a=X(i,:);
a=a';
e=theta'*a;
prediction=sigmoid(e);
sum=sum+((prediction-y(i))*a);
end
sum=sum/m;
grad=sum+((lambda/m)*theta);
end
end