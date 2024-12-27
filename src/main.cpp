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
    mlp.initWeights(MLP::InitMethod::UNIFORM_RANDOM);
    mlp.initBias(MLP::InitMethod::UNIFORM_RANDOM, -1, 1);
    mlp.setOutputLayerAct(MLP::ActFunc::RELU);
    mlp.setHiddenLayerAct(MLP::ActFunc::SIGMOID);
    mlp.setLR(0.01);

    mlp.miniBatchGD(data, targets, 100000);
    return 0;
}

int trainMNIST() {


    DataMNIST data = importMNIST(); // Import the Data (img, row, col)
    Uint8Vector3D imgData = data.imgs;
    std::vector<uint8_t> labelData = data.labels;
    std::vector<double> doubleLabels = castTargetsFromUint8ToDouble(labelData);
    DoubleVector2D oneHotLabels = oneHotEncodeTargets(doubleLabels);

    Uint8Vector2D flatData = Flatten3DTensor(imgData); // Flatten the data (pixel, img)
    DoubleVector2D flatDoubleData = castImgsFromUint8ToDouble(flatData); // Cast to Double
    normaliseData(flatDoubleData); // Normalise (z-score)


    // Split Data
    DoubleVector2D trainData= sliceCols(flatDoubleData, 0, 1000);
    DoubleVector2D trainLabels = sliceCols(oneHotLabels, 0, 1000);

    // Create mlp object
    std::vector<int> myStruct = { 784, 100, 32, 10};
    MLP mlp(myStruct);

    // Initialise Weights, Bias and HyperParams
    /* mlp.initWeights(MLP::InitMethod::UNIFORM_RANDOM, -0.1, 0.1); */
    /* mlp.initWeights(MLP::InitMethod::UNIFORM_RANDOM, -1, 1); */
    mlp.initBias(MLP::InitMethod::UNIFORM_RANDOM, -0.1, 0.1);
    DoubleVector3D myW = mlp.getWeights();
    printMatrix(myW[1]);
    return 0;
    mlp.setHiddenLayerAct(MLP::ActFunc::RELU);
    mlp.setOutputLayerAct(MLP::ActFunc::SOFTMAX);
    mlp.setLR(0.01);
    mlp.setBatchSize(32);
    mlp.setLossFunc(MLP::LossFunc::ENTROPY);

    // Exploration of where Nan comes from...
    /* ForwardPropResult res = mlp.forwardProp(sliceCols(trainData, 0, 3)); */
    /* DoubleVector2D outpLayer = res.a[2]; */
    /* printMatrix(outpLayer); */
    /* return 0; */

    /* std::string_view modelPath = "models/model2.bin"; */
    /* mlp.loadModel(modelPath); */
    mlp.miniBatchGD(trainData, trainLabels, 100);
    /* mlp.saveModel(modelPath); */
    /* return 0; */

    /* mlp.printModelInfo(); */

    ForwardPropResult result = mlp.forwardProp(sliceCols(trainData, 0, 12));
    DoubleVector2D outp = result.a[2];
    printMatrix(outp);

    std::vector<int> preds(outp[0].size());
    // Find Max idx for each example
    for (size_t colIdx = 0; colIdx < outp[0].size(); colIdx++) {
        double RunningMax = -INFINITY;
        for (size_t rowIdx = 0; rowIdx < outp.size(); rowIdx++) {
            if (outp[rowIdx][colIdx] > RunningMax) {
                RunningMax = outp[rowIdx][colIdx];
                preds[colIdx] = rowIdx;
            }
        }
    }

    std::string choice;
    for (size_t imgIdx = 0; imgIdx < preds.size(); imgIdx++) {

        printMNISTImg(imgData[imgIdx], 128);
        std::cout << std::endl << "Preditiction: "<< preds[imgIdx] << std::endl;
        std::cout << "Label: " << static_cast<int>(labelData[imgIdx]) << std::endl;
        std::cout << std::endl << "Press any key to view the next image, or 'e' to exit" << std::endl;
        std::cin >> choice;

        if (choice == "e") {
            break;
        }
    }
    /* imgData = buildImgFromFlat(flatData); */

    return 0;
}




