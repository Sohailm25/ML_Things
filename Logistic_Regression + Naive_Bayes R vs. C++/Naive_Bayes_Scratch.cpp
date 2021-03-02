
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <typeinfo>
#include <chrono>
#include <math.h>
// #include <armadillo>


using namespace std;

/* COEFFICIENTS FROM LogisticRegression in R:

(Intercept)      pclass 
   1.297166   -0.779929 
   
*/

struct Dataset {
        vector<int> pclass;
        vector<int> survived;
        vector<int> sex;
        vector<double> age;
        double count;
    };

int const TRAIN_SPLIT = 900;
int const TEST_SPLIT = 146; 

vector<double> sigmoid(vector<double> sigmoidInput);
void readCSV(ifstream &MyFile, vector<int> &pclass, vector<int> &survived, vector<int> &sex, vector<double> &age, double &count);
void trainTestSplit(Dataset &titanic, Dataset &trainTitanic, Dataset &testTitanic, int trainSplit, int testSplit);
vector<double> matrixVecMult(Dataset trainTitanic, vector<vector<double> > matrix, vector<double> vector, bool isTransverse=false);
double calc_age_lh (double v, double mean_v, double var_v);
double calc_raw_prob (int pclass, int sex, double age);
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

    std::chrono::time_point<std::chrono::system_clock> start, end; 
    

    // get the count of survived and perished
    double numSurvived = 0, numPerished = 0;
    double priorSurvived = 0, priorPerished = 0; 

    for (int i = 0; i < trainTitanic.count ; i++) {
        if (trainTitanic.survived[i] == 0) {
            numPerished++;
        }
        else if (trainTitanic.survived[i] == 1) {
            numSurvived++;
        }
    }

    vector<double> priors(2,0);
    priorSurvived = numSurvived / trainTitanic.count;
    priorPerished = numPerished / trainTitanic.count;
    priors[0] = priorPerished;
    priors[1] = priorSurvived;

    cout << "Num Survived vs Perished: " << numSurvived << " " << numPerished << endl;
    cout << "Prior Probabilities: " << priorSurvived << " " << priorPerished << endl;



    vector<vector<double> > lh_pclass, lh_sex, lh_age;
    lh_pclass.resize(2, vector<double>(3, 0));
    lh_sex.resize(2, vector<double>(2, 0));
    lh_age.resize(2, vector<double>(2, 0));

    /* PCLASS LIKELIHOOD */
    // count for pclass vs survived
    for (int i=0 ; i < trainTitanic.count ; i++) {
        for (int j=0 ; j < 2 ; j++) { // perished vs survived
            for (int k=1 ; k < 4; k++) { // pclass 1,2,3
                if ( (trainTitanic.pclass[i] == k) && (trainTitanic.survived[i] == j) ) {
                    lh_pclass[j][k-1]++;
                }
            }
        }
    }

    // likelihood for pclass
    for (int i=0 ; i < 2 ; i++) { // perished vs survived
        for (int j=0 ; j < 3; j++) { // pclass 1,2,3
            if (i == 0) { // if perished
                lh_pclass[i][j] = lh_pclass[i][j] / numPerished;
            }
            else if (i == 1) { // if survived
                lh_pclass[i][j] = lh_pclass[i][j] / numSurvived;
            }
        }
    }
    
    cout << "PCLASS LIKELIHOODS: " << lh_pclass[1][0] << " " << lh_pclass[1][1] << " " << lh_pclass[1][2] << endl;

    /* SEX LIKELIHOOD */
    // count for sex vs survived
    for (int i=0 ; i < trainTitanic.count ; i++) {
        for (int j=0 ; j < 2 ; j++) { // perished vs survived
            for (int k=0 ; k < 2; k++) { // sex 0,1
                if ( (trainTitanic.sex[i] == k) && (trainTitanic.survived[i] == j) ) {
                    lh_sex[j][k]++;
                }
            }
        }
    }    

    // likelihood for sex
    for (int i=0 ; i < 2 ; i++) { // perished vs survived
        for (int j=0 ; j < 2; j++) { // sex 0,1
            if (i == 0) { // if perished
                lh_sex[i][j] = lh_sex[i][j] / numPerished;
            }
            else if (i == 1) { // if survived
                lh_sex[i][j] = lh_sex[i][j] / numSurvived;
            }
        }
    }    

    cout << "SEX LIKELIHOODS: " << lh_sex[0][0] << " " << lh_sex[0][1] << endl;
    
    /* AGE LIKELIHOOD */
    // likelihood for age (mean/var + gaussian)
    vector<double> sumAge, countAge, agesSurvived, agesPerished;
    double sumAgeSurvived = 0, sumAgePerished=0, countAgeSurvived=0, countAgePerished=0;
    for (int i=0; i < trainTitanic.count; i++) {
        if (trainTitanic.survived[i] == 0) { // perished
            agesPerished.push_back(trainTitanic.age[i]);
            sumAgePerished += trainTitanic.age[i];
            countAgePerished++;
        }
        else if (trainTitanic.survived[i] == 1) { // survived
            agesSurvived.push_back(trainTitanic.age[i]);
            sumAgeSurvived += trainTitanic.age[i];
            countAgeSurvived++;
        }
    }

    sumAge.push_back(sumAgePerished);
    sumAge.push_back(sumAgeSurvived);
    countAge.push_back(countAgePerished);
    countAge.push_back(countAgeSurvived);

    vector<double> age_mean(2,0);
    vector<double> age_var(2,0);
    // calculate mean
    for (int i=0 ; i < 2; i++) {
        age_mean[i] = sumAge[i] / countAge[i];
    }
    // calculate variance
    long double sumSquaredPerished =0, sumSquaredSurvived =0;
    for (int i=0; i < agesPerished.size(); i++) {
        sumSquaredPerished += pow((agesPerished[i] - age_mean[0]),2);
        //cout << "(age - mean)^2 = " << pow((agesPerished[i] - age_mean[0]),2) << endl; 
        //cout << "FOR AGE: " << agesPerished[i] << " VARIANCE IS: " << sumSquaredPerished << endl;
    }
    for (int i=0; i< agesSurvived.size(); i++) {
        sumSquaredSurvived += pow((agesSurvived[i] - age_mean[1]), 2);
    }
    //cout << "SumSquaredPerished" << sumSquaredPerished << endl;
    //cout << "Size of AgesPerished" << agesPerished.size() << endl;

    age_var[0] = sumSquaredPerished / agesPerished.size();
    age_var[1] = sumSquaredSurvived / agesSurvived.size();


    cout << "MEAN AND VAR PERISHED: " << age_mean[0] << " " << age_var[0] << endl;
    cout << "MEAN AND VAR SURVIVED: " << age_mean[1] << " " << age_var[1] << endl;

    
    // bayes theorem

    double num_s, num_p, denominator, raw;
    vector<double> classification;
    double tp=0, tn=0, fp=0, fn=0;

    start = std::chrono::system_clock::now();
    // iterate through titanicdataset 
    for (int i=0 ; i < testTitanic.count; i++) {
        double pclassVar = testTitanic.pclass[i];
        double sexVar = testTitanic.sex[i];
        double ageVar = testTitanic.age[i];
        // cout << "\n pclassVar, sexVar, and ageVar = " << pclassVar << " " << sexVar << " " << ageVar << endl;
        vector<double> rawProbs(2,0);

        // cout << "lh_pclass: " << lh_pclass[1][pclassVar - 1] << endl;
        // cout << "lh_sex: " << lh_sex[1][sexVar] << endl;
        // cout << "priors[1]: " << priors[1] << endl;
        // cout << "calc_age_lh: " << calc_age_lh(ageVar, age_mean[1], age_var[1]) << endl;


        num_s = lh_pclass[1][pclassVar - 1] * lh_sex[1][sexVar] * priors[1] * calc_age_lh(ageVar, age_mean[1], age_var[1]);
        num_p = lh_pclass[0][pclassVar - 1] * lh_sex[0][sexVar] * priors[0] * calc_age_lh(ageVar, age_mean[0], age_var[0]);
        denominator = num_s + num_p;

        // cout << "num_s: " << num_s << endl;
        // cout << "num_p: " << num_p << endl;
    
        rawProbs[0] = num_p / denominator;
        rawProbs[1] = num_s / denominator;


        if (i < 5) {
            cout << "RAWPROBS FOR OBS " << i << ": " << rawProbs[0] << " " << rawProbs[1] << endl; 
        }

        if (rawProbs[1] > 0.5) {
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
    }

    end = std::chrono::system_clock::now();

    chrono::duration<double> elapsed_sec = end - start;
    cout << "RUNTime: " << elapsed_sec.count() << endl;

    double accuracy, sensitivity, specificity;

    accuracy = (tp + tn) / (tp + tn + fp + fn);
    sensitivity = (tp) / (tp + fn);
    specificity = (tn) / (tn + fp);

    cout << "\nMETRICS -> \nAccuracy: " << accuracy << "\nSensitivity: " << sensitivity << "\nSpecificity: " << specificity << endl;
    

    return 0;
}

double calc_age_lh (double v, double mean_v, double var_v) {
    return ( (1 / (sqrt(2 *  M_PI * var_v))) * exp(-pow((v - mean_v),2) / (2 * var_v)));
}

double calc_raw_prob (int pclass, int sex, double age) {

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

void readCSV(ifstream &MyFile, vector<int> &pclass, vector<int> &survived, vector<int> &sex, vector<double> &age, double &count) {

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