#include "mlp.h"
#include "helperfuncs.h"
#include <stdexcept>

void testGettersSetters();
int main() {


    /* std::string possibleDegrees[] = { "bachelor", "highschool" }; */
    /* return 0; */
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
    normaliseData(data);
    std::vector<double> target = separateTarget(data);

    std::vector<int> myStruct = { 5, 3, 2};
    MLP mlp(myStruct);

    mlp.initWeights(MLP::InitMethod::UNIFORM);
    mlp.initBias(MLP::InitMethod::UNIFORM, -10, 10);
    ForwardPropResult res = mlp.forwardProp(data);


    std::cout << "Output Layer" << std::endl;
    std::cout << "Num Rows: " << res.a[1].size() << std::endl;
    std::cout << "Num Cols: " << res.a[1][0].size() << std::endl;
    std::cout << std::endl << "Hidden Layer" << std::endl;
    std::cout << "Num Rows: " << res.a[0].size() << std::endl;
    std::cout << "Num Cols: " << res.a[0][0].size() << std::endl;
    std::cout << "Num Rows: " << res.z[0].size() << std::endl;
    std::cout << "Num Cols: " << res.z[0][0].size() << std::endl;
    /* mlp.backPropIteration(data, target); */
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
