
clear;

dpathProject = fullfile('/', 'home', 'mwang8', 'projects', 'def-jbpoline', 'mwang8', 'ukbb-cca');

dpathData = fullfile(dpathProject, 'data');
dpathToolbox = fullfile(dpathProject, 'toolbox');
dpathResults = fullfile(dpathProject, 'results');

dpathCCA = fullfile(dpathResults, 'cca');

dpathDataClean = fullfile(dpathData, 'clean');

fpathDataBehavioural = fullfile(dpathDataClean, 'behavioural_clean.csv');
fpathDataBrain = fullfile(dpathDataClean, 'brain_clean.csv');
fpathDataDemographic = fullfile(dpathDataClean, 'demographic_clean.csv');

fpathHoldout = fullfile(dpathDataClean, 'udi_holdout.csv');

fpathResultsCCA = fullfile(dpathCCA, 'cca.mat');
fpathResultsRegression = fullfile(dpathCCA, 'regression.mat');

save('paths.mat');
