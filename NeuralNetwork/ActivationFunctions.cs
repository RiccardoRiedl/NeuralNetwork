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
    /// passed from neurons. It returns the resulting activation value as well
    /// as the derivative to be used later in backpropagation
    /// </summary>
    /// <param name="type">Type of activation function to apply</param>
    /// <param name="input">Input values to activate</param>
    /// <param name="gradients">Whether to compute gradients/derivatives</param>
    /// <returns>Tuple containing activations and optional derivatives</returns>
    internal static (double[] activations, double[]? derivations) Activation(FunctionType type, double[] input, bool gradients)
    {
        double[] activations = new double[input.Length];
        double[]? derivations = gradients ? new double[input.Length] : null;

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
                derivations = gradients ? activations.Select(a => 1 - a * a).ToArray() : null;
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
                    // For SoftMax with cross-entropy loss, the gradient is typically handled
                    // in the loss function. For general case, SoftMax derivative is:
                    // dS_i/dx_j = S_i * (delta_ij - S_j)
                    // Since we typically use this with cross-entropy, we store the diagonal elements
                    derivations = activations.Select(a => a * (1 - a)).ToArray();
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
