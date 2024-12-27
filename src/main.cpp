#include "mlp.h"
#include "helperfuncs.h"
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string_view>

void testGettersSetters();
void testTemplateLossFuncs();
int trainSalarayData();
int trainMNIST();
int predictMNIST();

int main() {

    return predictMNIST();
    return trainMNIST();
    return trainSalarayData();
    return 0;
}

void testGettersSetters() {
    std::vector<int> myStruct = { 3, 10, 5};
    MLP mlp(myStruct);

    // Weights: TODO
    //
    // hidden layer act 

    std::cout << std::endl;
    std::cout << "Hidden Layer Activation Testing" << std::endl;
    printMlpEnum(mlp.getHiddenLayerAct());
    mlp.setHiddenLayerAct(MLP::ActFunc::ELU);
    printMlpEnum(mlp.getHiddenLayerAct());

    // Output layer act 
    std::cout << std::endl;
    std::cout << "Output Layer Activation Testing" << std::endl;
    printMlpEnum(mlp.getOutputLayerAct());
    mlp.setOutputLayerAct(MLP::ActFunc::ELU);
    printMlpEnum(mlp.getOutputLayerAct());

    // loss func 
    std::cout << std::endl;
    std::cout << "Loss Func Testing" << std::endl;
    printMlpEnum(mlp.getLossFunc());
    mlp.setLossFunc(MLP::LossFunc::ENTROPY);
    printMlpEnum(mlp.getLossFunc());

    // LR 
    std::cout << std::endl;
    std::cout << "Learning Rate Testing" << std::endl;
    std::cout << mlp.getLR() << std::endl;
    mlp.setLR(0.4);
    std::cout << mlp.getLR() << std::endl;

    // DecayRate 
    std::cout << std::endl;
    std::cout << "Decay Rate Testing" << std::endl;
    std::cout << mlp.getDecayRate() << std::endl;
    mlp.setDecayRate(90);
    std::cout << mlp.getDecayRate() << std::endl;

    // batchSize 
    std::cout << std::endl;
    std::cout << "Batch Size Testing" << std::endl;
    std::cout << mlp.getBatchSize() << std::endl;
    mlp.setBatchSize(10);
    std::cout << mlp.getBatchSize() << std::endl;
}

void testTemplateLossFuncs() {

    std::vector<int> myStruct;
    myStruct = { 5, 10, 3 };
    MLP mlp(myStruct);

    /* IntVector2D target(3, std::vector<int>(3)); */
    /* IntVector2D preds(3, std::vector<int>(3)); */
    DoubleVector2D target(3, std::vector<double>(3));
    DoubleVector2D preds(3, std::vector<double>(3));
    /* FloatVector2D target(3, std::vector<float>(3)); */
    /* FloatVector2D preds(3, std::vector<float>(3)); */

    target[0][0] = 3;
    target[1][0] = 4;
    target[2][0] = 5;
    target[0][1] = 6;
    target[1][1] = 7;
    target[2][1] = 8;
    target[0][2] = 9;
    target[1][2] = 10;
    target[2][2] = 11;

    preds[0][0] = 4;
    preds[1][0] = 5;
    preds[2][0] = 5;
    preds[0][1] = 6;
    preds[1][1] = 8;
    preds[2][1] = 9;
    preds[0][2] = 8;
    preds[1][2] = 12;
    preds[2][2] = 15;

    double loss;
    loss = mlp.calcLoss(target, preds);


    std::cout << loss << std::endl;

    std::cout << std::endl << std::endl;
    DoubleVector2D lossGradient;
    lossGradient = mlp.lossGradient(target, preds);

    for (const std::vector<double>& row : lossGradient) {
        for (const double& elem : row) {
            std::cout << elem << std::endl;
        }
    }
}


int trainSalarayData() {

    // Import data
    DoubleVector2D importedData;
    try {
        importedData = importCSV("data/CPSSW04.csv");
    } catch (const std::invalid_argument& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    } catch (const std::ios::ios_base::failure& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    normaliseData(importedData);

    // Separate data from targets
    std::vector<int> targetCols = { 5 };
    DoubleVector2D targets;
    try{
        targets = transpose(separateTarget(importedData, targetCols));
    } catch (const std::invalid_argument& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    DoubleVector2D data = transpose(importedData);

    // Create mlp object
    std::vector<int> myStruct = { 5, 25, 5, 1};
    MLP mlp(myStruct);

    // Initialise Weights, Bias and HyperParams
    /* double */ 
    /* mlp.initWeights(MLP::InitMethod::UNIFORM_RANDOM); */
    /* mlp.initBias(MLP::InitMethod::UNIFORM_RANDOM, -1, 1); */
    /* mlp.setOutputLayerAct(MLP::ActFunc::RELU); */
    /* mlp.setHiddenLayerAct(MLP::ActFunc::SIGMOID); */
    /* mlp.setLR(0.01); */

    mlp.miniBatchGD(data, targets, 100000);
    return 0;
}

int trainMNIST() {

    std::cout << "\nImporting MNIST Dataset..." << std::endl;

    // Preparing Data / Pre-proccessing
    DataMNIST* data = new DataMNIST(importMNIST()); 
    DoubleVector2D* flatDoubleData = new DoubleVector2D(castImgsFromUint8ToDouble(Flatten3DTensor(data->imgs)));
    DoubleVector2D* targets = new DoubleVector2D(castImgsFromUint8ToDouble(oneHotEncodeTargets(data->labels)));
    delete data;

    normaliseData(*flatDoubleData); // Normalise (z-score)
    
    // Split Data
    DoubleVector2D trainData= sliceCols(*flatDoubleData, 0, 9000);
    DoubleVector2D trainLabels = sliceCols(*targets, 0, 9000);
    delete flatDoubleData;
    delete targets;

    // Create mlp object
    std::vector<int> myStruct = { 784, 256, 128, 64, 10};
    MLP mlp(myStruct);

    // Initialise Weights, Bias and HyperParams
    double initParams[] = { 0, 0.01 };
    mlp.initWeights(MLP::InitMethod::GAUSSIAN_RANDOM, initParams);
    /* mlp.initBias(MLP::InitMethod::GAUSSIAN_RANDOM, params); */
    mlp.setHiddenLayerAct(MLP::ActFunc::RELU);
    mlp.setOutputLayerAct(MLP::ActFunc::SOFTMAX);
    mlp.setLR(0.001);
    mlp.setBatchSize(64);
    mlp.setLossFunc(MLP::LossFunc::ENTROPY);

    std::string_view modelPath = "models/model2.bin";
    /* mlp.loadModel(modelPath); */

    std::cout << "Now Training The Model: \n" << std::endl;
    mlp.printModelInfo(false);
    mlp.miniBatchGD(trainData, trainLabels, 100);
    mlp.saveModel(modelPath);
    return 0;
}

int predictMNIST() {

    // Import Data
    DataMNIST data = importMNIST(); 
    Uint8Vector3D imgData = data.imgs;
    std::vector<uint8_t> labelData = data.labels;

    // One Hot encode Data
    Uint8Vector2D oneHotLabels = oneHotEncodeTargets(labelData);
    // Make image data compatable with MLP
    DoubleVector2D* flatDoubleData = new DoubleVector2D();
    *flatDoubleData = castImgsFromUint8ToDouble(Flatten3DTensor(imgData)); 
    normaliseData(*flatDoubleData);

    // Init MLP, this is trivial since we loading one
    std::vector<int> myStruct = { 1, 1};
    MLP mlp(myStruct);
    std::string_view modelPath = "models/model2.bin";
    mlp.loadModel("models/model2.bin");

    /* mlp.printModelInfo(); */

    /* return 0; */
    /* printDims(*flatDoubleData); */
    /* return 0; */
    std::vector<uint8_t> preds = mlp.predict(*flatDoubleData);
    delete flatDoubleData;

    displayPredsMNIST(imgData, labelData, preds);
    return 0;
}


