%%%%%%%%%%%%%%%%
%
%   Machine Intelligence I: excercise 9.3
%
%	C-SVM
%
%%%%%%%%%%%%%%%%

% creating data

clear all

%rmpath
%addpath 'F:\Studies\WiSe1516\Machine Intelligence I\excercise_09\libsvm-3.21\'

%%create training set
for i = 1:80
    C1(i,:) = 0.5*(normrnd([0,1],0.1)+normrnd([1,0],0.1));
end
for i = 1:80
    C2(i,:) = 0.5*(normrnd([0,0],0.1)+normrnd([1,1],0.1));
end

dat = [C1;C2];
label = [ones(80,1);-ones(80,1)];
dat = [dat,label];
ind = randperm(length(dat));
dat = dat(ind,:);
clear ind

%% create test set
for i = 1:80
    C1(i,:) = 0.5*(normrnd([0,1],0.1)+normrnd([1,0],0.1));
end
for i = 1:80
    C2(i,:) = 0.5*(normrnd([0,0],0.1)+normrnd([1,1],0.1));
end

dat_test = [C1;C2];
label = [ones(80,1);-ones(80,1)];
dat_test = [dat_test,label];
ind = randperm(length(dat_test));
dat_test = dat_test(ind,:);
clear ind

%% parameter initialization / grid definition
C = [2^3,2^5,2^7];
gamma = [2,2^-1,2^-3];
index = 1:length(dat);

%% LOO - cross validation grid search for parameter optimization
% 9.4 a)

for g = 1:length(gamma)
    
    for c = 1:length(C)
        
        % LOOCV
        for i = 1:length(dat)
            CVind = index(index~=i);
            train = dat(CVind,1:2);
            model = svmtrain(train,label(CVind),'ShowPlot',false,'kernel_function','rbf','boxconstraint',C(c),'rbf_sigma',gamma(g),'boxconstraint',C(c));
            
            SVM(i) = svmclassify(model,dat(i,1:2),'Showplot',false);
        end
        perf(c,g) = length(find(SVM'~=label))/length(label);
        
    end
end


%% 9.4 b
contour(perf)

%9.4 c)
[row,col] = find(perf==max(max(perf)));

model = svmtrain(dat_test(:,1:2),label,'ShowPlot',true,'kernel_function','rbf','rbf_sigma',gamma(col),'boxconstraint', C(row));

        
