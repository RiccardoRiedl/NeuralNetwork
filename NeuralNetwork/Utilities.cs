using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using System.Xml.Linq;

namespace NeuralNetwork
{
    public static class Utilities
    {
        private static Random random = new Random();

        internal static void ThrowIfMaxSmallerMin(double min, double max)
        {
            if (min >= max)
            {
                throw new ArgumentException("Parameter max is smaller than min.");
            }
        }

        internal static double Random(double min, double max)
        {
            ThrowIfMaxSmallerMin(min, max);
            ArgumentNullException.ThrowIfNull(random);

            return random.NextDouble() * (max - min) + min;
        }

        internal static int IndexOfMax(double[] array)
        {
            double maxValue = array.Max();
            return Array.IndexOf(array, maxValue);
        }

        internal static double[] NormalizeArray(int[] array, double min = 0.0, double max = 1.0)
        {
            double[] doubleArray = array.Select(x => (double)x).ToArray();
            return NormalizeArray(doubleArray, min, max);
        }

        internal static double[] NormalizeArray(double[] array, double min = 0.0, double max = 1.0)
        {
            double[] normalizedArray = new double[array.Length];
            double range = max - min;

            // Find the minimum and maximum values in the array
            double currentMin = array.Min();
            double currentMax = array.Max();

            return array.Select(x => min + ((x - currentMin) / (currentMax - currentMin) * range)).ToArray();
        }
    }
}
