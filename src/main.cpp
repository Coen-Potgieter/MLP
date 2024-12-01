#include "mlp.h"
#include "helperfuncs.h"
#include <stdexcept>

void testGettersSetters();
void testTemplateLossFuncs();
int main() {

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
    std::vector<int> myStruct = { 5, 20, 10, 1};
    MLP mlp(myStruct);

    // Initialise Weights and Bias
    mlp.initWeights(MLP::InitMethod::UNIFORM);
    mlp.initBias(MLP::InitMethod::UNIFORM, -10, 10);

    // TODO: when moving this to acutal function make sure this is inside try catch
    DoubleVector2D data32 = sliceCols(data, 0, 31);
    DoubleVector2D target32 = sliceCols(targets, 0, 31);

    mlp.singleBackPropItter(data32, target32);
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
    lossGradient = mlp.avgLossGradient(target, preds);
    
    for (const std::vector<double>& row : lossGradient) {
        for (const double& elem : row) {
            std::cout << elem << std::endl;
        }
    }
}









