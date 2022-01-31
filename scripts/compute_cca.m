
%% parameters

nComponents1 = 25; % PCA components for brain data
nComponents2 = 25; % PCA components for behavioural data
nPerm = 100; % number of permutations for bootstrapping
nFold = 5; % for k-fold cross-validation
nRep = 15000; % number of repetitions of the k-fold cross-validation

%% load path variables

load('paths.mat', 'dpathToolbox', ...
    'fpathDataBehavioural', 'fpathDataBrain', 'fpathDataDemographic', 'fpathHoldout', ...
    'fpathResultsCCA', 'fpathResultsRegression')

%% print parameters

disp('----- Parameters -----');
disp(['nComponents1: ' num2str(nComponents1)]);
disp(['nComponents2: ' num2str(nComponents2)]);
disp(['nPerm: ' num2str(nPerm)]);
disp(['nFold: ' num2str(nFold)]);
disp(['nRep: ' num2str(nRep)]);
disp(['dpathToolbox: ' dpathToolbox]);
disp(['fpathDataBehavioural: ' fpathDataBehavioural]);
disp(['fpathDataBrain: ' fpathDataBrain]);
disp(['fpathDataDemographic: ' fpathDataDemographic]);
disp(['fpathHoldout: ' fpathHoldout]);
disp(['fpathResultsCCA: ' fpathResultsCCA]);
disp(['fpathResultsRegression: ' fpathResultsRegression]);
disp('----------------------');

%% load toolbox
addpath(genpath(dpathToolbox));

%% load data

tableBehavioural = loadData(fpathDataBehavioural);
tableBrain = loadData(fpathDataBrain);
tableDemographic = loadData(fpathDataDemographic);

udisBehavioural = tableBehavioural.Properties.VariableNames';
udisBrain = tableBrain.Properties.VariableNames';
udisDemographic = tableDemographic.Properties.VariableNames';

disp(['Behavioural data: ' num2str(height(tableBehavioural)) ', ' num2str(width(tableBehavioural))]);
disp(['Brain data: ' num2str(height(tableBrain)) ', ' num2str(width(tableBrain))]);
disp(['Demographic data: ' num2str(height(tableDemographic)) ', ' num2str(width(tableDemographic))]);
disp('--------------------');

%% remove holdout variable

tableHoldout = readtable(fpathHoldout);

if height(tableHoldout) ~= 1
    error('Cannot handle more than one holdout variable');
else
    udiHoldout = tableHoldout.('udi'){1};
end

if any(strcmp(udiHoldout, udisBehavioural))
    disp('Removing holdout variable from behavioural data');
    holdoutData = tableBehavioural(:, udiHoldout);
    tableBehavioural = removevars(tableBehavioural, udiHoldout);
elseif any(strcmp(udiHoldout, udisBrain))
    disp('Removing holdout variable from brain data');
    holdoutData = tableBrain(:, udiHoldout);
    tableBrain = removevars(tableBrain, udiHoldout);
elseif any(strcmp(udiHoldout, udisDemographic))
    disp('Removing holdout variable from demographic data');
    holdoutData = tableDemographic(:, udiHoldout);
    tableDemographic = removevars(tableDemographic, udiHoldout);
else
    error('Holdout variable not found in data tables');
end

disp('--------------------');

%% CCA

% estimate the cross-validated CCA
[ dat, cca ] = ccaFullKAnalysis( ...
    tableBrain{:,:}, tableBehavioural{:,:}, tableDemographic{:,:}, ...
    udisBrain, udisBehavioural, udisDemographic, udisBehavioural, ...
    nComponents1, nComponents2, nPerm, nFold, nRep, 'median');

save(fpathResultsCCA, 'dat', 'cca');

%% regression

% estimate the fit of a holdout variable to the primary canonical axis
[ R, S, pval ] = ccaLinRegCorr(cca, 1, holdoutData{:,:}, 1000);

save(fpathResultsRegression, 'R', 'S', 'pval');

%% helper functions

function data = loadData(fpath)

    data = readtable(fpath, ...
        'ReadRowNames', true, 'VariableNamingRule', 'preserve');
end
