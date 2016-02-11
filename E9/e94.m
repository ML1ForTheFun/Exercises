%%%%%%%%%%%%%%%%
%
%   Machine Intelligence I: excercise 9.3
%
%	C-SVM
%
%%%%%%%%%%%%%%%%

% creating data

clear all
close all

%rmpath
%addpath 'F:\Studies\WiSe1516\Machine Intelligence I\excercise_09\libsvm-3.21\'

%%create training set
std = .33;
for i = 1:80
   if randi(2)-1
      C1(i,:) = normrnd([0,1],std);
   else
      C1(i,:) = normrnd([1,0],std);
   end
end
for i = 1:80
   if randi(2)-1
      C2(i,:) = normrnd([0,0],std);
   else
      C2(i,:) = normrnd([1,1],std);
   end
end

dat = [C1;C2];
label = [ones(80,1);-ones(80,1)];
dat = [dat,label];
ind = randperm(length(dat));
dat = dat(ind,:);
clear ind

%% create test set
for i = 1:80
   if randi(2)-1
      C1(i,:) = normrnd([0,1],std);
   else
      C1(i,:) = normrnd([1,0],std);
   end
end
for i = 1:80
   if randi(2)-1
      C2(i,:) = normrnd([0,0],std);
   else
      C2(i,:) = normrnd([1,1],std);
   end
end

dat_test = [C1;C2];
label = [ones(80,1);-ones(80,1)];
dat_test = [dat_test,label];
ind = randperm(length(dat_test));
dat_test = dat_test(ind,:);
clear ind

%% parameter initialization / grid definition
C = [2^-5,2^-3,2^-1,2^1,2^3,2^5,2^7,2^9];
gamma = [2^-15,2^-13,2^-11,2^-9,2^-7,2^-5,2^-3,2^-1,2^1];
index = 1:length(dat);

%% LOO - cross validation grid search for parameter optimization
% 9.4 a)
for g = 1:length(gamma)
    
    for c = 1:length(C)
        
        % LOOCV
        for i = 1:length(dat)
            CVind = index(index~=i);
            train = dat(CVind,1:2);
            model = svmtrain(train,dat(CVind,3),'ShowPlot',false,'kernel_function','rbf','boxconstraint',C(c),'rbf_sigma',gamma(g));
            
            trainsetclass = svmclassify(model,dat(CVind,1:2),'Showplot',false);
            trainperf(i) = sum(trainsetclass==dat(CVind,3))/length(CVind);
            SVM(i) = svmclassify(model,dat(i,1:2),'Showplot',false);
        end
        meantrainclass(c, g) = mean(trainperf);
        CVperf(c,g) = length(find(SVM'==dat(:,3)))/length(dat(:,3));
        
        fprintf('C: %d of %d, gamma: %d of %d\n', c, length(C), g, length(gamma))
    end
end

%% 9.4 b
figure
ContourCVperf = contour(gamma,C,CVperf);
title('9.4.b - Cross Validation Performance');
xlabel('gamma');
ylabel('C');
clabel(ContourCVperf)

set(gca,'XScale','log')
set(gca,'YScale','log')

figure
clabel(contour(gamma,C,meantrainclass));
xlabel('gamma');
ylabel('C');
title('9.4.b - Mean Training Classification');

set(gca,'XScale','log')
set(gca,'YScale','log')

%9.4 c)
[row,col] = find(CVperf==max(max(CVperf)),1);

%9.4 d)
figure
model = svmtrain(dat(:,1:2),dat(:,3),'ShowPlot',true,'kernel_function','rbf','rbf_sigma',gamma(col),'boxconstraint', C(row));
title('9.4.d - Optimal C, gamma classification');
        
%9.4 e)
SVM = svmclassify(model,dat_test(:,1:2),'Showplot',true);
error = sum(SVM~=dat_test(:,3))/length(dat_test);
fprintf('Classification error: %f \nNumber of Support Vectors: %d \n', error, size(model.SupportVectors, 1))

