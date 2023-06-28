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
    public static List<TrainingData> LoadFromFile(string fileName)
    {
        StreamReader r = new StreamReader(fileName);
        string json = r.ReadToEnd();
        var res = JsonSerializer.Deserialize<List<TrainingData>>(json);
        return res;
    }

    public static void SaveDataSet(List<TrainingData> data, string fileName)
    {
        string jsonString = JsonSerializer.Serialize(data, new JsonSerializerOptions() { WriteIndented = true });
        using (StreamWriter outputFile = new StreamWriter(fileName))
        {
            outputFile.WriteLine(jsonString);
        }
    }
}
