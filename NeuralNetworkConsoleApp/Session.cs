using NeuralNetwork;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkConsoleApp;

public sealed class Session
{
    private static readonly Lazy<Session> lazy = new Lazy<Session>(() => new Session());

    public static Session Instance { get { return lazy.Value; } }

    private Session()
    {
    }

    public void AddTrainingData(TrainingData trainingData)
    {
        if (CurrentTrainingData == null)
        {
            CurrentTrainingData = new List<TrainingData>();
        }

        CurrentTrainingData.Add(trainingData);
    }

    public LayeredNetwork? CurrentNetwork { get; set; }
    public List<TrainingData>? CurrentTrainingData { get; set; }
}
