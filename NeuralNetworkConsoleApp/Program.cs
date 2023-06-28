using NeuralNetwork;
using System.Collections.Specialized;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text;
using static System.Runtime.InteropServices.JavaScript.JSType;

internal class Program
{
    public sealed class Sessions
    {
        private static readonly Lazy<Sessions> lazy = new Lazy<Sessions>(() => new Sessions());

        public static Sessions Instance { get { return lazy.Value; } }

        private Sessions()
        {
        }

        public LayeredNetwork? CurrentNetwork { get; set; }
        public List<TrainingData>? CurrentTrainingData { get; set; }
    }

    private static void Main(string[] args)
    {
        Console.WriteLine("Neural Network Console App");

        PrintHelp();

        var key = SelectCommand();
        while (key != ConsoleKey.Escape)
        {
            try
            {
                switch (key)
                {
                    case ConsoleKey.F1:
                        PrintHelp();
                        break;

                    case ConsoleKey.C:
                        CreateNetwork();
                        break;

                    case ConsoleKey.P:
                        PrintNetwork();
                        break;

                    case ConsoleKey.F:
                        FeedNetwork();
                        break;

                    case ConsoleKey.T:
                        TrainNetwork();
                        break;

                    case ConsoleKey.S:
                        SaveData();
                        break;

                    case ConsoleKey.L:
                        LoadData();
                        break;

                    case ConsoleKey.D:
                        PrintData();
                        break;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine();
                Console.WriteLine("Error: " + ex.Message);
            }

            key = SelectCommand();
        }

        Console.Write("Press any key to close this window");
        Console.ReadKey();
    }


    /// <summary>
    /// Helper to query for the next command
    /// </summary>
    /// <returns></returns>
    static ConsoleKey SelectCommand()
    {
        Console.Write("\nSelect command: ");
        var key = Console.ReadKey().Key;
        Console.WriteLine();
        return key;
    }

    /// <summary>
    /// Prints an overview of all commands
    /// </summary>
    static void PrintHelp()
    {
        Console.WriteLine();
        Console.WriteLine("Press [F1] for help");
        Console.WriteLine("Press [c] to create a new layered network");
        Console.WriteLine("Press [p] to print a previously created network");
        Console.WriteLine("Press [f] to feed values into the network");
        Console.WriteLine("Press [t] to train the network through back propagation");
        Console.WriteLine("Press [l] to load training data from a json file");
        Console.WriteLine("Press [s] to save training data to a json file");
        Console.WriteLine("Press [a] to add training data");
        Console.WriteLine("Press [d] to print the current training data");
        Console.WriteLine("Press [Esc] to exit");
    }

    /// <summary>
    /// Command to create a new network
    /// </summary>
    static void CreateNetwork()
    {
        string prompt = "Enter array of layers with count of neurons (e.g. [10, 4, 4, 2])";
        var layers = ReadIntArray(prompt);
        Sessions.Instance.CurrentNetwork = new LayeredNetwork(layers, true);
    }

    /// <summary>
    /// Print network to console
    /// </summary>
    static void PrintNetwork()
    {
        ArgumentNullException.ThrowIfNull(Sessions.Instance.CurrentNetwork);
        Console.WriteLine(Sessions.Instance.CurrentNetwork.ToString());
    }

    /// <summary>
    /// Command to feed values into the network and print the result
    /// </summary>
    static void FeedNetwork()
    {
        ArgumentNullException.ThrowIfNull(Sessions.Instance.CurrentNetwork);

        string prompt = "Enter input values as array of doubles ([0.1, 0.25, ...])";
        var input = ReadDoubleArray(prompt);

        if (input.Length != Sessions.Instance.CurrentNetwork.InputCount)
        {
            Console.WriteLine("Number of input values does not match networks input nodes. Try again...");
            FeedNetwork();
        }

        var result = Sessions.Instance.CurrentNetwork.FeedForward(input, false);
        Console.WriteLine("Result: " + ArrayToString(result));
    }

    /// <summary>
    /// Command to train the network
    /// </summary>
    static void TrainNetwork()
    {
        ArgumentNullException.ThrowIfNull(Sessions.Instance.CurrentNetwork);
        ArgumentNullException.ThrowIfNull(Sessions.Instance.CurrentTrainingData);

        var lr = ReadDouble("Enter learning rate");

        int i = 0;
        Stopwatch stopwatch = Stopwatch.StartNew();

        for (;i < Sessions.Instance.CurrentTrainingData.Count;i++)
        {
            Sessions.Instance.CurrentNetwork.BackPropagation(Sessions.Instance.CurrentTrainingData[i], 0.01);
        }
        stopwatch.Stop();

        Console.WriteLine($"Training finished {i} runs after {stopwatch.ElapsedMilliseconds} ms.");
    }

    static void PrintData()
    {
        ArgumentNullException.ThrowIfNull(Sessions.Instance.CurrentNetwork);
        ArgumentNullException.ThrowIfNull(Sessions.Instance.CurrentTrainingData);

        foreach(var set in Sessions.Instance.CurrentTrainingData)
        {
            Console.WriteLine($"Input: {ArrayToString(set.Input)} -> Target: {ArrayToString(set.Target)}");
        }
    }

    static void AddTrainingData()
    {
        //ArgumentNullException.ThrowIfNull(network);
    }








    static string GetFilePath(bool throwIfDoesntExist, bool createIfNotExist)
    {
#if DEBUG
        return @"C:\Users\adriri1\Desktop\TestFiles\New\test.json";
#endif
        Console.WriteLine();
        Console.Write("Enter file: ");

        var path = Console.ReadLine();

        if (!File.Exists(path))
        {
            if (throwIfDoesntExist)
            {
                throw new FileNotFoundException(path);
            }

            if (createIfNotExist)
            {
                var dir = Path.GetDirectoryName(path);
                if (!Directory.Exists(dir))
                {
                    Directory.CreateDirectory(dir);
                }
                File.Create(path).Close();
            }
        }

        return path;
    }

    static void SaveData()
    {
        var path = GetFilePath(false, true);

        //trainingData.Clear();
        //trainingData.Add(new TrainingDataSet(new double[] { 1.0, 2 }, new double[] { 0.3, 0.4 }));
        //trainingData.Add(new TrainingDataSet(new double[] { 3, 4.0 }, new double[] { 0.5, 0.6 }));


        TrainingDataPersistence.SaveDataSet(Sessions.Instance.CurrentTrainingData, path);
    }

    static void LoadData()
    {
        var path = GetFilePath(true, false);

        var data = TrainingDataPersistence.LoadFromFile(path);
        Sessions.Instance.CurrentTrainingData = data;

        //trainingData = new List<TrainingDataSet>(Utilities.LoadFromFile(path));
    }



    static string ArrayToString(double[] array)
    {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < array.Length; i++)
        {
            if (i > 0)
            {
                sb.Append(", ");
            }
            sb.Append(array[i].ToString("N4"));
        }
        sb.Append("]");
        return sb.ToString();
    }


    static int[] ReadIntArray(string prompt)
    {
        Console.Write(prompt + ": ");

        string value = ReadString();
        string[] elements = value.Trim('[', ']').Split(',');
        int[] intArray = new int[elements.Length];

        for (int i = 0; i < elements.Length; i++)
        {
            intArray[i] = int.Parse(elements[i]);
        }

        return intArray;
    }

    static double ReadDouble(string prompt)
    {
        Console.Write(prompt + ": ");
        string value = ReadString();

        double result;
        if (!double.TryParse(value, out result))
        {
            Console.WriteLine("Invalid input. Please try again...");
            return ReadDouble(prompt);
        }

        return result;
    }

    static double[] ReadDoubleArray(string prompt)
    {
        Console.Write(prompt + ": ");

        string value = ReadString();
        string[] elements = value.Trim('[', ']').Split(',');
        double[] array = new double[elements.Length];

        for (int i = 0; i < elements.Length; i++)
        {
            array[i] = double.Parse(elements[i]);
        }

        return array;
    }

    static string ReadString()
    {
        ConsoleKeyInfo keyInfo;
        string input = string.Empty;

        while (true)
        {
            keyInfo = Console.ReadKey();

            if (keyInfo.Key == ConsoleKey.Escape)
            {
                throw new Exception("Escape key pressed");
            }

            else if (keyInfo.Key == ConsoleKey.Backspace)
            {
                // Remove the last character from the input string (if any)
                if (input.Length > 0)
                {
                    input = input.Substring(0, input.Length - 1);
                    Console.Write("\b \b"); // Erase the character from the console
                }
            }
            else if (keyInfo.Key == ConsoleKey.Enter)
            {
                break; // Exit the loop when the user presses Enter
            }
            else
            {
                // Append the entered character to the input string
                input += keyInfo.KeyChar;
            }
        }
        return input;
    }
}