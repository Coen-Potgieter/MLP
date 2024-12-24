#include "mlp.h"
#include "helperfuncs.h"
#include <cstdint>
#include <stdexcept>
#include <string_view>

void testGettersSetters();
void testTemplateLossFuncs();
int trainSalarayData();
int trainMNIST();

int main() {
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

    DoubleVector2D loss;
    loss = mlp.calcLoss(target, preds);
    
    for (const std::vector<double>& row : loss) {
        for (const double& elem : row) {
            std::cout << elem << std::endl;
        }
    }

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
    mlp.initWeights(MLP::InitMethod::UNIFORM);
    mlp.initBias(MLP::InitMethod::UNIFORM, -1, 1);
    mlp.setOutputLayerAct(MLP::ActFunc::RELU);
    mlp.setHiddenLayerAct(MLP::ActFunc::SIGMOID);
    mlp.setLR(0.01);

    mlp.miniBatchGD(data, targets, 100000);
    return 0;
}

int trainMNIST() {


    /* DataMNIST data = importMNIST(); // Import the Data (img, row, col) */
    /* Uint8Vector3D imgData = data.imgs; */
    /* std::vector<uint8_t> labelData = data.labels; */
    /* std::vector<double> doubleLabels = castTargetsFromUint8ToDouble(labelData); */
    /* DoubleVector2D oneHotLabels = oneHotEncodeTargets(doubleLabels); */

    /* Uint8Vector2D flatData = Flatten3DTensor(imgData); // Flatten the data (pixel, img) */
    /* DoubleVector2D flatDoubleData = castImgsFromUint8ToDouble(flatData); // Cast to Double */
    /* normaliseData(flatDoubleData); // Normalise (z-score) */
    

    /* // Split Data */
    /* DoubleVector2D trainData= sliceCols(flatDoubleData, 0, 1000); */
    /* DoubleVector2D trainLabels = sliceCols(oneHotLabels, 0, 1000); */

    // Create mlp object
    std::vector<int> myStruct = { 10, 5, 7, 2, 10};
    MLP mlp(myStruct);

    
    // Initialise Weights, Bias and HyperParams
    mlp.initWeights(MLP::InitMethod::UNIFORM);
    mlp.initBias(MLP::InitMethod::UNIFORM, -1, 1);

    std::string_view modelPath = "models/test.bin";
    mlp.saveModel(modelPath);
    mlp.loadModel(modelPath);
    return 0;

    // How to set Enum using integers
    mlp.setHiddenLayerAct(static_cast<MLP::ActFunc>(0));

/*     DoubleVector3D weights = mlp.getWeights(); */
/*     std::cout << weights.size() << std::endl; */
/*     /1* printDims(const DoubleVector2D &mat) *1/ */
/*     // Test Saving of weights */
/*     return 0; */




/*     mlp.setOutputLayerAct(MLP::ActFunc::RELU); */
/*     mlp.setHiddenLayerAct(MLP::ActFunc::SIGMOID); */
/*     mlp.setLR(0.1); */

/*     mlp.miniBatchGD(trainData, trainLabels, 1000); */


  



    return 0;
    /* size_t imgIdx = 1; */
    /* imgData = buildImgFromFlat(flatData); */
    /* printMNISTImg(imgData[imgIdx], 128); */
    /* std::cout << static_cast<int>(labelData[imgIdx]) << std::endl; */
}




