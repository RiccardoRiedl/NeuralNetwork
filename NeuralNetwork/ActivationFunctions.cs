namespace NeuralNetwork;

/// <summary>
/// To be used within networks to assign activation function
/// </summary>
public enum FunctionType
{
    /// <summary>
    /// Sigmoid Activation Function
    /// </summary>
    Sigmoid,

    /// <summary>
    /// Rectified Linear Unit (ReLU) Activation Function
    /// </summary>
    ReLU,

    /// <summary>
    /// Hyperbolic Tangent (tanh) Activation Function
    /// </summary>
    Tanh,

    /// <summary>
    /// Leaky ReLU Activation Function
    /// </summary>
    LeakyReLU,

    /// <summary>
    /// Linear Activation Function
    /// </summary>
    Linear,

    /// <summary>
    /// SoftMax Function
    /// </summary>
    SoftMax
}

internal static class ActivationFunctions
{
    internal static double LeakyReLUAlpha = 0.01;


    /// <summary>
    /// Function to run an activation function based on a given input value
    /// passed from neurons. It retuns the resulting activation value as well
    /// as the derivation to be used later in backpropagation
    /// </summary>
    /// <param name="type"></param>
    /// <param name="input"></param>
    /// <param name="gradients"></param>
    /// <returns></returns>
    internal static (double[] activations, double[]? derivations) Activation(FunctionType type, double[] input, bool gradients)
    {
        double[] activations = new double[input.Length];
        double[]? derivations = new double[input.Length];

        switch (type)
        {
            case FunctionType.Sigmoid:
                activations = input.Select(x => 1 / (1 + Math.Exp(-x))).ToArray();
                derivations = gradients ? activations.Select(x => x * (1 - x)).ToArray() : null;
                break;

            case FunctionType.ReLU:
                activations = input.Select(x => x >= 0 ? x : 0).ToArray();
                derivations = gradients ? input.Select(x => x >= 0 ? 1.0 : 0).ToArray() : null;
                break;

            case FunctionType.Tanh:
                activations = input.Select(x => Math.Tanh(x)).ToArray();
                derivations = gradients ? input.Select(x => 1 - Math.Tanh(x) * Math.Tanh(x)).ToArray() : null;
                break;

            case FunctionType.LeakyReLU:
                activations = input.Select(x => x >= 0 ? x : LeakyReLUAlpha * x).ToArray();
                derivations = gradients ? input.Select(x => x >= 0 ? 1 : LeakyReLUAlpha).ToArray() : null;
                break;

            case FunctionType.Linear:
                activations = input.Select(x => x).ToArray();
                derivations = gradients ? input.Select(x => 1.0).ToArray() : null;
                break;

            case FunctionType.SoftMax:
                double maxInput = input.Max();
                double[] expInput = input.Select(x => Math.Exp(x - maxInput)).ToArray();
                double sumExpInput = expInput.Sum();
                activations = expInput.Select(x => x / sumExpInput).ToArray();

                if (gradients)
                {
                    // Compute the elements of the Jacobian matrix
                    double[,] jacobian = new double[input.Length, input.Length];
                    for (int i = 0; i < input.Length; i++)
                    {
                        for (int j = 0; j < input.Length; j++)
                        {
                            double delta = (i == j) ? 1.0f : 0.0f;
                            jacobian[i, j] = activations[i] * (delta - activations[j]);
                        }
                    }

                    // Compute the derivative of the softmax function with respect to the input
                    for (int i = 0; i < input.Length; i++)
                    {
                        double sum = 0.0f;
                        for (int j = 0; j < input.Length; j++)
                        {
                            sum += jacobian[i, j] * input[j];
                        }
                        derivations[i] = sum;
                    }
                }
                else
                {
                    derivations = null;
                }
                break;
        }

        return (activations, derivations);
    }

    /*
     * Additional
     * 
     * static public float Softplus(float x) => (float)Math.Log(1.0f + Math.Exp(x));
     * 
     * static public float ArcTan(float x) => (float)Math.Atan(x);
     * 
     * static public float BinaryStep(float x) => x < 0.0f ? 0.0f : 1.0f;
     */
}
