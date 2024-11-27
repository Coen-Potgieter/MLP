#include "mlp.h"
#include "helperfuncs.h"
#include <stdexcept>

void testGettersSetters();
int main() {

    DoubleVector2D data;
    try {

        data = importCSV("data/CPSSW04.csv");
    } catch (const std::invalid_argument& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    } catch (const std::ios::ios_base::failure& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    printData(data);

    return 0;
    std::vector<int> myStruct = { 3, 10, 5};
    MLP mlp(myStruct);

    /* std::vector<int> groundTruth = { 4, 1 }; */
    /* std::vector<int> results = { 9, 1 }; */
    std::vector<double> groundTruth = { 4.3, 1 };
    std::vector<double> results = { 9.3, 1 };
    double errors = mlp.calcAvgLoss(groundTruth, results);

    std::cout << errors << std::endl;
    return 0;
    DoubleVector2D myInput = { {1,2,3,5}, {4,5,6,7}, {7,8,9,10} };

    /* MLP mlp(myStruct); */
    mlp.initWeights(MLP::InitMethod::UNIFORM);
    mlp.initBias(MLP::InitMethod::UNIFORM, -10, 10);

    // test constructor 

/*     std::cout << mlp.getLR() << std::endl; */
     

    // test getters and setters
    if (mlp.getLossFunc() == MLP::LossFunc::MSE) {
        std::cout << "MSE" << std::endl;
    } else if (mlp.getLossFunc() == MLP::LossFunc::ENTROPY) {
        std::cout << "ENTROPY" << std::endl;
    }

    mlp.setLossFunc(MLP::LossFunc::ENTROPY);
    if (mlp.getLossFunc() == MLP::LossFunc::MSE) {
        std::cout << "MSE" << std::endl;
    } else if (mlp.getLossFunc() == MLP::LossFunc::ENTROPY) {
        std::cout << "ENTROPY" << std::endl;
    }
    /* std::cout << mlp.getLR() << std::endl; */

    ForwardPropResult res = mlp.forwardProp(myInput);

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
