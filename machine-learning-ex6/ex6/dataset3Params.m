function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1.0;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

Cvec = [0.01 0.03 0.1 0.3 1 3 10 30];
sigmaVec = Cvec;

finalC = 1;
finalsig = 1;
minError = 1000;

for i = 1:length(Cvec)
    for j = 1: length(sigmaVec)
        C = Cvec(i);
        sigma = sigmaVec(j);
        % Try different SVM Parameters here
        % [C, sigma] = dataset3Params(X, y, Xval, yval);

        % Train the SVM
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        % visualizeBoundary(X, y, model);
        pred = svmPredict(model, Xval);
        error = mean(double(pred ~= yval));
        fprintf('C = %f, sigma = %f, error = %f\n', C, sigma, error);
        if(error < minError)
            finalC = C;
            finalsig = sigma;
            minError = error;
        end
    end
end

C = finalC;
sigma = finalsig;




% =========================================================================

end
