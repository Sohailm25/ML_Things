
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <typeinfo>
#include <chrono>
// #include <armadillo>


using namespace std;
using namespace std::chrono;

/* COEFFICIENTS FROM LogisticRegression in R:

(Intercept)      pclass 
   1.297166   -0.779929 
   
*/

struct Dataset {
        vector<int> pclass;
        vector<int> survived;
        vector<int> sex;
        vector<double> age;
        int count;
    };

int const TRAIN_SPLIT = 900;
int const TEST_SPLIT = 146; 

vector<double> sigmoid(vector<double> sigmoidInput);
void readCSV(ifstream &MyFile, vector<int> &pclass, vector<int> &survived, vector<int> &sex, vector<double> &age, int &count);
void trainTestSplit(Dataset &titanic, Dataset &trainTitanic, Dataset &testTitanic, int trainSplit, int testSplit);
vector<double> matrixVecMult(Dataset trainTitanic, vector<vector<double> > matrix, vector<double> vector, bool isTransverse=false);

int main() {

    // intialize variables
    ifstream MyFile("titanic_project.csv");
    Dataset titanic;


    Dataset trainTitanic;
    Dataset testTitanic;

    // populate vectors with readCSV function
    readCSV(MyFile, titanic.pclass, titanic.survived, titanic.sex, titanic.age, titanic.count);

    // split into train and test sets
    trainTestSplit(titanic, trainTitanic, testTitanic, TRAIN_SPLIT, TEST_SPLIT);
    
    // cout << "\nNumber of Total Observations: " << titanic.count;
    // cout << "\nNumber of Train Observations: " << trainTitanic.count;
    // cout << "\nNumber of Test Observations: " << testTitanic.count;
    // cout << "Observation 1 PCLASS : " << titanic.pclass[0] << " age: " << titanic.age[0];

    /* START THE CLOCK */

    // initialize a vector of size 2, all with values of 1 (intercept and pclass weights)
    int numCols = 2;
    int numRows = trainTitanic.count;
    
    vector<double> weights (2,1);

    // initialize the second column of the data_matrix to be the pclass of each observation
    vector<vector<double> > data_matrix;
    data_matrix.resize(numRows, vector<double>(numCols, 1));
    for (int i = 0; i < trainTitanic.count; i++) {
        data_matrix[i][1] = trainTitanic.pclass[i];
    }


    // cout << "Data Matrix 0 0 " << data_matrix[4][1];

    // create a labels vector for survived or not 
    vector<double> labels;
    labels.resize(trainTitanic.count, 3);
    for (int i = 0; i < trainTitanic.count; i++) {
        labels[i] = trainTitanic.survived[i];
    }

    // cout << "Observation 60 survived: " << labels[60];
    // cout << "Observation 61 survived: " << labels[61];
    // cout << "Observation 62 survived: " << labels[62];

    // converting to ANACONDA Matrices

    //mat labels_matrix = conv_to<mat>::from(labels);
    //mat data_matrix_2 = conv_to<mat>::from(data_matrix);



    /* 
        data_matrix -> 900 x 2
        weights     -> 1 x 2
        error       -> 900 x 1

        data_matrix -> ROWS = trainTitanic.count number of observations
        data_matrix -> COL0 = ALL 1's (to multiply against weights[0])
        data_matrix -> COL1 = trainTitanic.pclass pclass of every observation (to multiply against weights[1])
    
    */

    // data_matrix % * % weights
    
    // cout matrix

    // cout << "Observation 897: " << data_matrix[897][0] << data_matrix[897][1] << endl;
    // cout << "Observation 898: " << data_matrix[898][0] << data_matrix[898][1] << endl;
    // cout << "Observation 899: " << data_matrix[899][0] << data_matrix[899][1] << endl;
    // cout << "Observation 900: " << data_matrix[900][0] << data_matrix[900][1] << endl;
    // cout << "Observation 901: " << data_matrix[901][0] << data_matrix[901][1] << endl;


    auto start = high_resolution_clock::now();

    double const LEARNING_RATE = 0.001;
    int const ITERATIONS = 50000;
    double const EPSILON = .0000001;


    // GRADIENT DESCENT
    for (int i = 0; i < ITERATIONS; i++) {

        vector<double> sigmoidInput;
        vector<double> probVector;
        sigmoidInput = matrixVecMult(trainTitanic, data_matrix, weights);
        probVector = sigmoid(sigmoidInput);

        vector<double> error(TRAIN_SPLIT, 0);
        for (int i = 0; i < error.size() ; i++) {
            error[i] = labels[i] - probVector[i];
        }

        // dataError is a 2 x 1 vector
        vector<double> dataError;
        dataError = matrixVecMult(trainTitanic, data_matrix, error, true);

        double prevWeight1, prevWeight2;
        prevWeight1 = weights[0];
        prevWeight2 = weights[1];

        for (int i = 0; i < weights.size() ; i++) {
            weights[i] = weights[i] + ( LEARNING_RATE * dataError[i] );
        }
        
        double diff1 = (abs(prevWeight1 - weights[0]));
        double diff2 = (abs(prevWeight2 - weights[1]));


        if ((diff1 < EPSILON) && (diff2 < EPSILON)) {
            cout << "WEIGHTS AFTER GRADIENT DESCENT ITERATION " << i << ": " << weights[0] << " and " << weights[1] << endl;
            cout << " PROGRAM TERMINATED ON ITERATION " << i << " WITH WEIGHTS: " << weights[0] << " and " << weights[1] << endl;
            break;
        }

    } 

    auto stop = high_resolution_clock::now();

    // TEST DATA
    vector<double> probabilities;
    vector<double> classification;
    double fp = 0, fn =0, tp=0, tn=0;
    for (int i = 0 ; i < testTitanic.count; i++) {
        double logOdds = (testTitanic.pclass[i] * weights[1]) + weights[0];
        double prob = exp(logOdds) / (1 + exp(logOdds));
        probabilities.push_back(prob);
        double survivedClassification;
        if (prob > 0.5) {
            if (testTitanic.survived[i] == 1) {
                tp++;
            }
            if (testTitanic.survived[i] == 0) {
                fp++;
            }
            classification.push_back(1);
        }
        else {
            if (testTitanic.survived[i] == 1) {
                fn++;
            }
            if (testTitanic.survived[i] == 0) {
                tn++;
            }
            classification.push_back(0);
        }
        classification.push_back(survivedClassification);
        if (i < 10) {
            cout << "OBSERVATION " << i << " Survival Probability: " << prob << " and Classification: " << survivedClassification << endl;
        }
    }

    double accuracy, sensitivity, specificity;

    accuracy = (tp + tn) / (tp + tn + fp + fn);
    sensitivity = (tp) / (tp + fn);
    specificity = (tn) / (tn + fp);

    cout << "\n METRICS -> \nAccuracy: " << accuracy << "\nSensitivity: " << sensitivity << "\nSpecificity: " << specificity << endl;


    

    chrono::duration<double> elapsed_sec = stop - start;
    cout << "TOTAL RUNTime: " << elapsed_sec.count() << endl;


    return 0;
}

vector<double> matrixVecMult(Dataset trainTitanic, vector<vector<double> > matrix, vector<double> vector, bool isTransverse) {

    std::vector<double> solution;

    // cout << isTransverse;

    // outer is the rows/observations of the data_matrix
    if (!isTransverse) {
        for (int i = 0; i < trainTitanic.count ; i++) {
            // inner is the columns of data_matrix and weights vector 
            double temp = 0;
            for (int j = 0; j < 2; j++) {
                double mult = 0;

                double val1 = matrix[i][j];
                double val2 = vector[j];

                //cout << "Matrix Val1: " << val1 << endl;
                //cout << "Vector Val2: " << val2 << endl;

                mult = val1 * val2;

                //cout << "Multiplied Value: " << mult << endl;

                temp += mult;
            }

            //cout << "Value being added to solution vector: " << temp << endl;
            solution.push_back(temp);
        }
    }
    else {
        for (int i = 0; i < 2 ; i++) {
            // inner is the columns of data_matrix and weights vector 
            double temp = 0;
            for (int j = 0; j < trainTitanic.count; j++) {
                double mult = 0;

                double val1 = matrix[j][i];
                double val2 = vector[j];

                //cout << "Matrix Val1: " << val1 << endl;
                //cout << "Vector Val2: " << val2 << endl;

                mult = val1 * val2;

                //cout << "Multiplied Value: " << mult << endl;

                temp += mult;
            }

            //cout << "Value being added to solution vector: " << temp << endl;
            solution.push_back(temp);
        }

        
    }

    return solution;
}


vector<double> sigmoid(vector<double> sigmoidInput) {

    vector<double> sigmoidSolution;

    for (int i = 0; i < sigmoidInput.size() ; i++) {
        double temp = 0;
        temp += ( (1.0) / (1.0 + exp(-sigmoidInput[i])) );
        sigmoidSolution.push_back(temp);
    }

    return sigmoidSolution;
}

void readCSV(ifstream &MyFile, vector<int> &pclass, vector<int> &survived, vector<int> &sex, vector<double> &age, int &count) {

    count = 0;

    string trash;
    // trash first line bc column names
    getline(MyFile, trash);

    // Write the CSV File into Vectors for each column
    while(MyFile.good())
    { 
        count++;
        string observationNum;
        string pclassCurrent;
        string survivedCurrent;
        string sexCurrent;
        string ageCurrent;

        getline(MyFile, observationNum, ',');
        getline(MyFile, pclassCurrent, ',');
        getline(MyFile, survivedCurrent, ',');
        getline(MyFile, sexCurrent, ',');
        getline(MyFile, ageCurrent);
        
        try {
            int pclassInt = stoi(pclassCurrent);
            int survivedInt = stoi(survivedCurrent);
            int sexInt = stoi(sexCurrent);
            double ageDouble = stod(ageCurrent);
            
            pclass.push_back(pclassInt);
            survived.push_back(survivedInt);
            sex.push_back(sexInt);
            age.push_back(ageDouble);
        }
        catch(const std::invalid_argument) {
            continue;
        }
    }
}

void trainTestSplit(Dataset &titanic, Dataset &trainTitanic, Dataset &testTitanic, int trainSplit, int testSplit) {

    // initialize train and test dataset values
    trainTitanic.pclass.resize(trainSplit, 0);
    trainTitanic.survived.resize(trainSplit, 0);
    trainTitanic.sex.resize(trainSplit, 0);
    trainTitanic.age.resize(trainSplit, 0);
    testTitanic.pclass.resize(testSplit, 0);
    testTitanic.survived.resize(testSplit, 0);
    testTitanic.sex.resize(testSplit, 0);
    testTitanic.age.resize(testSplit, 0);

    int countTrain = 0;
    int countTest = 0;

    // training data
    for (int i = 0; i < trainSplit; i++) {
        countTrain++;
        trainTitanic.pclass[i] = titanic.pclass[i];
        trainTitanic.survived[i] = titanic.survived[i];
        trainTitanic.sex[i] = titanic.sex[i];
        trainTitanic.age[i] = titanic.age[i];
    }
    trainTitanic.count = countTrain;

    // test data
    for (int i = 0; i < testSplit; i++) {
        countTest++;
        testTitanic.pclass[i] = titanic.pclass[trainSplit + i];
        testTitanic.survived[i] = titanic.survived[trainSplit + i];
        testTitanic.sex[i] = titanic.sex[trainSplit + i];
        testTitanic.age[i] = titanic.age[trainSplit + i];
    }
    testTitanic.count = countTest;



}