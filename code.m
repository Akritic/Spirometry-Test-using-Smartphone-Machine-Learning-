%------------------------------------------------------------------------%
%                             ABOUT THE CODE                             %
%------------------------------------------------------------------------%
% Description: MATLAB SCRIPT TO CALCULATE THE PARAMETERS OF SPIROMETRY
%------------------------------------------------------------------------%
clc;clear;close all;
%------------------------------------------------------------------------%

A = 0.001e-6; % m^2 / Male Adult Hood

filenames = {...
            'female.wav'
%             'gymgirl1.wav'
%             'gymgirl2.wav'
%             'male.wav'
%             'Name.wav'
%             'Name11.wav'
%             'ritikaexhale_normal.wav'
%             'ritikastairs.wav'
%             'ritstairs.wav'
%             'sabarinormal.wav'
%             'saumyanormalARM.wav'
%             'saumyanormalNEAR.wav'
%             'shubham_big_mouthed_exhale.wav'
%             'shubhamstairs.wav'
%             'test1.wav'
%             'test2.wav'
%             'try1breath.wav'
%             'try2lessdistbreath.wav'
%             'whisle2.wav'
%             'whistle1.wav'
            };

numFiles = length(filenames);

fprintf('--------------------------------------------------------------------------------------\n')
fprintf('                  Name of File     FEV1     FVC    FEV1/FVC \n')
fprintf('--------------------------------------------------------------------------------------\n')
for i = 1:numFiles

[audio,Fs] = audioread(filenames{i});

audio = audio(:,1);
n = length(audio);
t = 0:1/Fs:(n-1)/Fs;

figure()
plot(t,audio)
title(sprintf('File: %s',filenames{i}))
grid on
axis square
xlabel('Time [sec]')
ylabel('y(t)')

Ts = 1/Fs;
segsize = 1e-3;
interval = round(segsize/Ts);

count = 0;
for j = 0:interval:length(t)-interval
    
    count = count + 1;
    seg(count,:)  = audio(j+1:j+interval);
    time(count,:) = t(j+1:j+interval);
    
    low  = 100;
    high = 1200;
    [yf(count,:),b(count,:),a(count,:)] = apply_filter(time(count,:),...
                                   seg(count,:),[low high]);
    
    Yf(count,:) = fft(yf(count,:));                           
    x(count,1) = mean(real(Yf(count,:)));

    if count == 5
        figure
        plot_spectrum(time(count,:),seg(count,:))
        xlabel('Frequency Hz')
        ylabel('Y(f)')
        title(sprintf('Frequency response for a segment in %s',filenames{i}))
    end
    
end

    Y = -0.000229.*x.^2 + 0.0442*x + 1.002;
    FR = Y.*A.*1e3;
    
FEV1 = trapz(Y(1:1000));
FVC  = trapz(Y); 
ratio = FEV1/FVC;    

TimeFR = linspace(0,n*Ts,length(Y));

figure()
plot(TimeFR,FR)
axis square
grid on
xlabel('Time [s]')
ylabel('Flow Rate [litre/sec]')
title('Flow Rate vs Time')

fprintf('%30s   %5.2f  %5.2f  %5.2f \n',filenames{i},FEV1,FVC,ratio)

save(sprintf('%s_segments.mat',filenames{i}),'seg');
save(sprintf('%s_times.mat',filenames{i}),'time');
save(sprintf('%s_Mean.mat',filenames{i}),'x');
save(sprintf('%s_FlowRate.mat',filenames{i}),'Y');

clear seg time yf Yf x Y FR

end
fprintf('======================================================================================\n')
%------------------------------------------------------------------------%
% Perform classification


fprintf('--------------------------------------------------------------------------------------\n')
fprintf('            Name of File       Training Times          Mean Squared Error  \n')
fprintf('--------------------------------------------------------------------------------------\n')

for i = 1:numFiles

load(sprintf('%s_Mean.mat',filenames{i}));
load(sprintf('%s_FlowRate.mat',filenames{i}));

Id_train = round(0.7*length(x));

xtrain = x(1:Id_train);
ytrain = Y(1:Id_train);

xtest = x(Id_train+1:end);
ytest = Y(Id_train+1:end);

for ii=1:Id_train
   if Y(ii)>mean(ytrain)
      svmtraindata(ii,1) = 1;
   else
      svmtraindata(ii,1) = 0;
   end
end


tic;
SVM  = fitcsvm(xtrain,svmtraindata);
ttsvm = toc;

tic;
KNN  = fitcknn(xtrain,ytrain);
ttknn = toc;

tic;
Tree = fitctree(xtrain,ytrain);
tttree = toc;

predict_svm = predict(SVM,xtest);
predict_knn = predict(KNN,xtest);
predict_tree = predict(Tree,xtest);

mse_svm  = mse(ytest,predict_svm);
mse_knn  = mse(ytest,predict_knn);
mse_tree = mse(ytest,predict_tree);

fprintf('%25s | %5.2f | %5.2f | %5.2f | %5.2e | %5.2e | %5.2e | \n',...
    filenames{i},ttsvm,ttknn,tttree,mse_svm,mse_knn,mse_tree)

clear xtrain ytrain xtest ytest svmtraindata
end

fprintf('======================================================================================\n')
%------------------------------------------------------------------------%