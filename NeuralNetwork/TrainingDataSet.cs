using System.Text.Json;

namespace NeuralNetwork;

/// <summary>
/// Data set for training neural network
/// </summary>
public struct TrainingData
{
    public TrainingData(double[] input, double[] target)
    {
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
