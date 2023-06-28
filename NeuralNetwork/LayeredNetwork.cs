using System.Text;

namespace NeuralNetwork;

/// <summary>
/// Neural network with fully connected layers and optionally randomized
/// weights and biases. Provides feed forward method for prediction and
/// back propagation implementation for training
/// </summary>
public class LayeredNetwork
{
    int[] layerSizes;           // Number of neurons in each layer
    FunctionType[] functions;   // Activation function for each layer
    int layerCount;             // Total number of layers
    double[][] layerOutputs;    // Activations of neurons in each layer
    double[][] layerDerivations;// Derviation of neurons in each layer
    double[][] layerInputs;     // Weighted sums of neurons in each layer
    double[][] biases;          // Weighted sums of neurons in each layer
    double[][][] weights;       // Weights between layers

    public int InputCount => layerSizes[0];
    public int OutputCount => layerSizes[layerCount - 1];

    /// <summary>
    /// Create a new neural network with fully connected layers
    /// </summary>
    /// <param name="layerSizes">Each element of the array represents a layer
    /// defined by the number of neurons</param>
    /// <param name="randomize">Optionally randomize weights and biases</param>
    public LayeredNetwork(int[] layerSizes, bool randomize)
    {
        this.layerSizes = layerSizes;
        this.layerCount = layerSizes.Length;

        // Create all the required arrays based on the layer count
        functions = new FunctionType[layerCount];
        layerOutputs = new double[layerCount][];
        layerDerivations = new double[layerCount][];
        layerInputs = new double[layerCount][];
        biases = new double[layerCount][];

        weights = new double[layerCount - 1][][];

        for (int i = 0; i < layerCount; i++)
        {
            layerOutputs[i] = new double[layerSizes[i]];
            layerDerivations[i] = new double[layerSizes[i]];
            layerInputs[i] = new double[layerSizes[i]];
            biases[i] = new double[layerSizes[i]];
            functions[i] = FunctionType.Sigmoid;
        }

        // Initialize weights
        for (int i = 0; i < layerCount - 1; i++)
        {
            weights[i] = new double[layerSizes[i + 1]][];
            for (int j = 0; j < layerSizes[i + 1]; j++)
            {
                weights[i][j] = new double[layerSizes[i]];

                if (randomize)
                {
                    for (int k = 0; k < layerSizes[i]; k++)
                    {
                        weights[i][j][k] = Utilities.Random(-0.5, 0.5);
                    }
                }
            }
        }

        for (int j = 0; j < layerSizes[0]; j++)
        {
            biases[0][j] = 0;
        }

        for (int i = 1; i < layerCount - 1; i++)
        {
            for (int j = 0; j < layerSizes[i]; j++)
            {
                biases[i][j] = randomize ? Utilities.Random(-0.5, 0.5) : 0;
            }
        }
    }

    /// <summary>
    /// Set activation function for one layer
    /// </summary>
    /// <param name="iLayer">Index of layer</param>
    /// <param name="type">Activiaton function type to set</param>
    public void SetFunc(int iLayer, FunctionType type)
    {
        if (iLayer >= layerCount || iLayer < 0)
        {
            throw new ArgumentException("Invalid layer index");
        }

        functions[iLayer] = type;
    }


    public double[] FeedForward(double[] input, bool gradients)
    {

        //if (NormalizeInput)
        //{
        //    layerOutputs[0] = Utilities.NormalizeArray(input);
        //}
        //else
        //{
            Array.Copy(input, layerOutputs[0], input.Length);
        //}

        for (int iSourceLayer = 0; iSourceLayer < layerCount - 1; iSourceLayer++)
        {
            int iTargetLayer = iSourceLayer + 1;

            for (int iTargetNeuron = 0; iTargetNeuron < layerSizes[iTargetLayer]; iTargetNeuron++)
            {
                // Start with the bias
                double weightedSum = biases[iTargetLayer][iTargetNeuron];

                // Iterate over all incoming connections and sum up the
                // products of the edges weights and respective value
                for (int iSourceNeuron = 0; iSourceNeuron < layerSizes[iSourceLayer]; iSourceNeuron++)
                {
                    weightedSum += weights[iSourceLayer][iTargetNeuron][iSourceNeuron] * layerOutputs[iSourceLayer][iSourceNeuron];
                }

                layerInputs[iTargetLayer][iTargetNeuron] = weightedSum;
            }

            var a = ActivationFunctions.Activation(functions[iTargetLayer], layerInputs[iTargetLayer], gradients);
            layerOutputs[iTargetLayer] = a.activations;
            layerDerivations[iTargetLayer] = (double[])a.derivations;
        }

        return layerOutputs[layerCount - 1];
    }

    /// <summary>
    /// Back propagation
    /// </summary>
    /// <param name="input"></param>
    /// <param name="target"></param>
    /// <param name="learningRate"></param>
    public double BackPropagation(TrainingData values, double learningRate)
    {
        double totalError = 0;

        // Perform forward propagation to calculate activations and weighted sums
        FeedForward(values.Input, true);

        // Calculate output layer error
        int iOutputSize = layerSizes[layerCount - 1];
        double[] outputError = new double[iOutputSize];
        for (int iOutputNeuron = 0; iOutputNeuron < iOutputSize; iOutputNeuron++)
        {
            outputError[iOutputNeuron] = layerOutputs[layerCount - 1][iOutputNeuron] - values.Target[iOutputNeuron];
            totalError += outputError[iOutputNeuron];
        }

        // Backpropagate the error
        double[][] hiddenErrors = new double[layerCount][];
        hiddenErrors[layerCount - 1] = outputError;

        for (int iLayer = layerCount - 2; iLayer >= 0; iLayer--)
        {
            // Just for readability
            int iPreviousLayer = iLayer + 1;

            hiddenErrors[iLayer] = new double[layerSizes[iLayer]];

            for (int iNeuron = 0; iNeuron < layerSizes[iLayer]; iNeuron++)
            {
                // "previous" means the one where the error comes from during
                // back propagation. This is actually the "next" layer, but
                // since we are going backwards, this naming makes more sense
                for (int iPreviousNeuron = 0; iPreviousNeuron < layerSizes[iPreviousLayer]; iPreviousNeuron++)
                {
                    hiddenErrors[iLayer][iNeuron] +=                        // we sum up all partial errors
                        weights[iLayer][iPreviousNeuron][iNeuron] *         // weight the impact of the error
                        layerDerivations[iPreviousLayer][iPreviousNeuron] * // gradient
                        hiddenErrors[iPreviousLayer][iPreviousNeuron];      // source error
                }
            }

            // Now update the biases and weights based on the hidden erros
            for (int iPreviousNeuron = 0; iPreviousNeuron < layerSizes[iPreviousLayer]; iPreviousNeuron++)
            {
                for (int iNeuron = 0; iNeuron < layerSizes[iLayer]; iNeuron++)
                {
                    weights[iLayer][iPreviousNeuron][iNeuron] -=
                        learningRate *
                        hiddenErrors[iPreviousLayer][iPreviousNeuron] *
                        layerDerivations[iPreviousLayer][iPreviousNeuron] *
                        layerOutputs[iLayer][iNeuron];
                }

                biases[iPreviousLayer][iPreviousNeuron] -=
                    learningRate *
                    hiddenErrors[iPreviousLayer][iPreviousNeuron] *
                    layerDerivations[iPreviousLayer][iPreviousNeuron];
            }
        }

        return totalError;
    }

    /// <summary>
    /// Print network
    /// </summary>
    /// <returns></returns>
    public override string ToString()
    {
        StringBuilder sb = new StringBuilder();

        sb.AppendLine($"{layerCount} Layers: => [{string.Join("] [", layerSizes)}]");
        for (int i = 0; i < weights.Length; i++)
        {
            for (int j = 0; j < weights[i].Length; j++)
            {
                for (int k = 0; k < weights[i][j].Length; k++)
                {
                    sb.AppendLine($"From layer {i} to {j} from {k}: Weight = {weights[i][j][k]}");
                }
            }
        }
        for (int i = 0; i < biases.Length; i++)
        {
            for (int j = 0; j < biases[i].Length; j++)
            {
                sb.AppendLine($"layer {i} neuron {j}: Bias  = {biases[i][j]}");
            }

        }

        Console.WriteLine($"");
        return sb.ToString();
    }
}