using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkConsoleApp;

internal class EscapeException : Exception
{
    public EscapeException() : base("Escape key pressed")
    {
    }
}
