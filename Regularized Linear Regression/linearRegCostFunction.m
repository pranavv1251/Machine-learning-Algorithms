function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
grad
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
sum=0;
for i=1:m
a=(X(i,:))';
z=((theta'*a)-y(i))^2;
sum=sum+z;
grad=grad+((theta'*a)-y(i))*a;
end
grad=grad/m;
J=(sum/m)/2;
sum=0;
for j=2:length(theta)
    sum=sum+(theta(j,1)^2);
    grad(j,1)=grad(j,1)+((lambda/m)*theta(j,1));
end
sum=(lambda/(2*m))*sum;
    
J=J+sum;




% =========================================================================

grad = grad(:);

end
