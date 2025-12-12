using System.Text.Json;

namespace NeuralNetwork;

/// <summary>
/// Data set for training neural network
/// </summary>
public struct TrainingData
{
    /// <summary>
    /// Create a new training data entry with input and target values
    /// </summary>
    /// <param name="input">Input values for the neural network</param>
    /// <param name="target">Expected target/output values</param>
    /// <exception cref="ArgumentNullException">If input or target is null</exception>
    /// <exception cref="ArgumentException">If arrays contain NaN or infinite values</exception>
    public TrainingData(double[] input, double[] target)
    {
        ArgumentNullException.ThrowIfNull(input, nameof(input));
        ArgumentNullException.ThrowIfNull(target, nameof(target));

        if (input.Length == 0)
        {
            throw new ArgumentException("Input array cannot be empty", nameof(input));
        }

        if (target.Length == 0)
        {
            throw new ArgumentException("Target array cannot be empty", nameof(target));
        }

        if (input.Any(double.IsNaN) || input.Any(double.IsInfinity))
        {
            throw new ArgumentException("Input contains NaN or infinite values", nameof(input));
        }

        if (target.Any(double.IsNaN) || target.Any(double.IsInfinity))
        {
            throw new ArgumentException("Target contains NaN or infinite values", nameof(target));
        }

        Input = input;
        Target = target;
    }

    public double[] Input { get; set; }
    public double[] Target { get; set; }
}

public static class TrainingDataPersistence
{
    /// <summary>
    /// Load training data from a JSON file
    /// </summary>
    /// <param name="fileName">Path to the JSON file</param>
    /// <returns>List of training data, or null if deserialization fails</returns>
    public static List<TrainingData>? LoadFromFile(string fileName)
    {
        using StreamReader r = new StreamReader(fileName);
        string json = r.ReadToEnd();
        return JsonSerializer.Deserialize<List<TrainingData>>(json);
    }

    /// <summary>
    /// Save training data to a JSON file
    /// </summary>
    /// <param name="data">Training data to save</param>
    /// <param name="fileName">Path to save the JSON file</param>
    public static void SaveDataSet(List<TrainingData> data, string fileName)
    {
        string jsonString = JsonSerializer.Serialize(data, new JsonSerializerOptions() { WriteIndented = true });
        using StreamWriter outputFile = new StreamWriter(fileName);
        outputFile.WriteLine(jsonString);
    }
}
