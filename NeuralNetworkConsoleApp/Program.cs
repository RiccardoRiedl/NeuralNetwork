using NeuralNetwork;
using NeuralNetworkConsoleApp;
using System.Diagnostics;
using System.Text;

internal class Program
{
    /// <summary>
    /// The main function is looping getting commands until
    /// the [Esc] key is pressed to exit the app
    /// </summary>
    /// <param name="args"></param>
    private static void Main(string[] args)
    {
        StartMessage();

        var key = SelectCommand();
        while (key != ConsoleKey.Escape)
        {
            try
            {
                ProcessCommand(key);
            }
            catch (Exception ex)
            {
                PrintError(ex);
            }

            key = SelectCommand();
        }

        Console.Write("Press any key to close this window");
        Console.ReadKey();
    }

    /// <summary>
    /// Print a title followed by the help content to start
    /// </summary>
    private static void StartMessage()
    {
        Console.WriteLine("-============================-");
        Console.WriteLine("| Neural Network Console App |");
        Console.WriteLine("| © 2025 Riccardo Riedl      |");
        Console.WriteLine("-============================-");
        Console.WriteLine();
        Console.WriteLine("Press [F1] for help");
        Console.WriteLine();
    }

    /// <summary>
    /// Clear the screen and print welcome message
    /// </summary>
    private static void ClearScreen()
    {
        Console.Clear();
        StartMessage();
    }

    /// <summary>
    /// Just mapping the input key to the methods to run the command
    /// Needs to be in sync with help
    /// </summary>
    /// <param name="key"></param>
    private static void ProcessCommand(ConsoleKey key)
    {
        ArgumentNullException.ThrowIfNull(key);

        switch (key)
        {
            case ConsoleKey.F1:
                PrintHelp();
                break;

            case ConsoleKey.Delete:
                ClearScreen();
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

            case ConsoleKey.L:
                LoadData();
                break;

            case ConsoleKey.S:
                SaveData();
                break;

            case ConsoleKey.A:
                AddData();
                break;

            case ConsoleKey.D:
                PrintData();
                break;
        }
    }

    /// <summary>
    /// Helper to query for the next command
    /// </summary>
    /// <returns></returns>
    static ConsoleKey SelectCommand()
    {
        Console.Write("Select command: ");
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
        Console.WriteLine("Press [F1]  for help");
        Console.WriteLine("Press [Del] to clear screen");

        Console.WriteLine("Press [c]   to create a new layered network");
        Console.WriteLine("Press [p]   to print the current network");
        Console.WriteLine("Press [f]   to feed values into the network");
        Console.WriteLine("Press [t]   to train the network through back propagation");
        Console.WriteLine("Press [l]   to load training data from a json file");
        Console.WriteLine("Press [s]   to save training data to a json file");
        Console.WriteLine("Press [a]   to add training data");
        Console.WriteLine("Press [d]   to print the current training data");

        Console.WriteLine("Press [Esc] to exit");
    }

    /// <summary>
    /// Command to create a new network
    /// </summary>
    static void CreateNetwork()
    {
        string prompt = "Enter array of layers with count of neurons (e.g. [10, 4, 4, 2])";
        var layers = ReadIntArray(prompt);

        Session.Instance.CurrentNetwork = new LayeredNetwork(layers, true);
    }

    /// <summary>
    /// Print network to console
    /// </summary>
    static void PrintNetwork()
    {
        ArgumentNullException.ThrowIfNull(Session.Instance.CurrentNetwork);
        Console.WriteLine(Session.Instance.CurrentNetwork.ToString());
    }

    /// <summary>
    /// Command to feed values into the network and print the result
    /// </summary>
    static void FeedNetwork()
    {
        ArgumentNullException.ThrowIfNull(Session.Instance.CurrentNetwork);

        string prompt = $"Enter {Session.Instance.CurrentNetwork.InputCount} input values as array of doubles ([0.1, 0.25, ...])";
        var input = ReadDoubleArray(prompt, Session.Instance.CurrentNetwork.InputCount);

        var result = Session.Instance.CurrentNetwork.FeedForward(input, false);
        Console.WriteLine($"{Environment.NewLine}Result: {ArrayToString(result)}");
    }

    /// <summary>
    /// Command to train the network
    /// </summary>
    static void TrainNetwork()
    {
        ArgumentNullException.ThrowIfNull(Session.Instance.CurrentNetwork);
        ArgumentNullException.ThrowIfNull(Session.Instance.CurrentTrainingData);

        var lr = ReadDouble("Enter learning rate");
        var epochs = (int)ReadDouble("Enter number of epochs");

        int totalIterations = 0;
        Stopwatch stopwatch = Stopwatch.StartNew();

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            for (int i = 0; i < Session.Instance.CurrentTrainingData.Count; i++)
            {
                Session.Instance.CurrentNetwork.BackPropagation(Session.Instance.CurrentTrainingData[i], lr);
                totalIterations++;
            }
        }
        stopwatch.Stop();

        Console.WriteLine($"Training finished {totalIterations} iterations ({epochs} epochs) after {stopwatch.ElapsedMilliseconds} ms.");
    }

    /// <summary>
    /// Prints the current training data
    /// </summary>
    static void PrintData()
    {
        ArgumentNullException.ThrowIfNull(Session.Instance.CurrentNetwork);
        ArgumentNullException.ThrowIfNull(Session.Instance.CurrentTrainingData);

        foreach (var set in Session.Instance.CurrentTrainingData)
        {
            Console.WriteLine($"Input: {ArrayToString(set.Input)} -> Target: {ArrayToString(set.Target)}");
        }
    }

    /// <summary>
    /// Add a new set of training data
    /// </summary>
    static void AddData()
    {
        ArgumentNullException.ThrowIfNull(Session.Instance.CurrentNetwork);

        var input = ReadDoubleArray($"Enter {Session.Instance.CurrentNetwork.InputCount} input values", Session.Instance.CurrentNetwork.InputCount);
        var output = ReadDoubleArray($"Enter {Session.Instance.CurrentNetwork.TargetCount} target values", Session.Instance.CurrentNetwork.TargetCount);

        Session.Instance.AddTrainingData(new TrainingData(input, output));
    }

    /// <summary>
    /// Gets the path for the json file to store training data
    /// </summary>
    /// <param name="throwIfDoesntExist"></param>
    /// <param name="createIfNotExist"></param>
    /// <returns></returns>
    /// <exception cref="FileNotFoundException"></exception>
    static string GetFilePath(bool throwIfDoesntExist, bool createIfNotExist)
    {
        var cur = Directory.GetCurrentDirectory();

        Console.WriteLine("Current directory where files are stored:");
        Console.WriteLine(cur);
        Console.Write("Enter file name: ");

        var file = Console.ReadLine();

        ArgumentNullException.ThrowIfNull(file);

        var path = Path.Combine(cur, file);
        path = Path.ChangeExtension(path, "json");

        if (!File.Exists(path))
        {
            if (throwIfDoesntExist)
            {
                throw new FileNotFoundException($"File not found: {path}");
            }

            if (createIfNotExist)
            {
                var dir = Path.GetDirectoryName(path);
                ArgumentNullException.ThrowIfNull(dir);

                if (!Directory.Exists(dir))
                {
                    Directory.CreateDirectory(dir);
                }
                File.Create(path).Close();
            }
        }

        return path;
    }

    /// <summary>
    /// Save current training data
    /// </summary>
    static void SaveData()
    {
        ArgumentNullException.ThrowIfNull(Session.Instance.CurrentTrainingData);
        var path = GetFilePath(false, true);
        TrainingDataPersistence.SaveDataSet(Session.Instance.CurrentTrainingData, path);
    }

    /// <summary>
    /// Load training data from file
    /// </summary>
    static void LoadData()
    {
        var path = GetFilePath(true, false);
        var data = TrainingDataPersistence.LoadFromFile(path);
        Session.Instance.CurrentTrainingData = data;
    }

    /// <summary>
    /// Converts array of doubles into a readable string like [1.2, 3.0, ... ]
    /// </summary>
    /// <param name="array"></param>
    /// <returns></returns>
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



    /// <summary>
    /// Read one double value
    /// </summary>
    /// <param name="prompt">Message to print</param>
    /// <returns>double value if successful</returns>
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

    /// <summary>
    /// Prompt user to enter an array of double values as "[x.y, z, ...]"
    /// </summary>
    /// <param name="prompt">Message to print</param>
    /// <param name="expectedLength">Expected number of elements</param>
    /// <returns>double array if successful</returns>
    static double[] ReadDoubleArray(string prompt, int expectedLength)
    {
        Console.Write($"{prompt}: ");

        string value = ReadString();
        string[] elements = value.Trim('[', ']').Split(',');

        // Check if expected number of elements was entered and if not restart the prompt
        if (elements.Length != expectedLength)
        {
            Console.WriteLine($"{Environment.NewLine}Invalid input. Expected {expectedLength} elements but received {elements.Length}");
            return ReadDoubleArray(prompt, expectedLength);
        }

        // Now create the array from the string elements
        double[] array = new double[elements.Length];

        for (int i = 0; i < elements.Length; i++)
        {
            array[i] = double.Parse(elements[i]);
        }

        Console.WriteLine();
        return array;
    }

    static int[] ReadIntArray(string prompt)
    {
        Console.Write($"{prompt}: ");

        string value = ReadString();
        string[] elements = value.Trim('[', ']').Split(',');
        int[] intArray = new int[elements.Length];

        for (int i = 0; i < elements.Length; i++)
        {
            intArray[i] = int.Parse(elements[i]);
        }

        Console.WriteLine();
        return intArray;
    }

    /// <summary>
    /// Custom ReadString that can be escaped with [Esc] to cancel an input command
    /// </summary>
    /// <returns>The input string</returns>
    /// <exception cref="EscapeException">When prompt is escaped</exception>
    static string ReadString()
    {
        ConsoleKeyInfo keyInfo;
        string input = string.Empty;

        while (true)
        {
            keyInfo = Console.ReadKey();

            switch (keyInfo.Key)
            {
                case ConsoleKey.Escape:
                    throw new EscapeException();

                case ConsoleKey.Enter:
                    return input;

                // Remove the last character from the input string (if any)
                case ConsoleKey.Backspace:
                    if (input.Length > 0)
                    {
                        input = input.Substring(0, input.Length - 1);
                        // Erase the character from the console
                        Console.Write(" \b");
                    }
                    break;

                // Append the entered character to the input string
                default:
                    input += keyInfo.KeyChar;
                    break;

            }
        }
    }

    /// <summary>
    /// Just print the message of the exception
    /// </summary>
    /// <param name="ex"></param>
    internal static void PrintError(Exception ex)
    {
        Console.WriteLine(" ");
        Console.WriteLine("Error: " + ex.Message);
        Console.WriteLine();
    }
}