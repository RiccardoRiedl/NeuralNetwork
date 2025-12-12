using System;
using System.Linq;

namespace NeuralNetwork
{
    public static class Utilities
    {
        internal static void ThrowIfMaxSmallerMin(double min, double max)
        {
            if (min >= max)
            {
                throw new ArgumentException("Parameter max is smaller than min.");
            }
        }

        /// <summary>
        /// Generate a random double value between min and max using thread-safe Random.Shared
        /// </summary>
        /// <param name="min">Minimum value (inclusive)</param>
        /// <param name="max">Maximum value (exclusive)</param>
        /// <returns>Random double value in the specified range</returns>
        internal static double Random(double min, double max)
        {
            ThrowIfMaxSmallerMin(min, max);
            return System.Random.Shared.NextDouble() * (max - min) + min;
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
