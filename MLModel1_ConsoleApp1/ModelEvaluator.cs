using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLModel1_ConsoleApp1
{
    public class ModelEvaluator
    {
        private readonly MLContext _mlContext;

        public ModelEvaluator()
        {
            _mlContext = new MLContext();
        }

        public void EvaluateModel(ITransformer model, IDataView testData)
        {
            var predictions = model.Transform(testData);
            var metrics = _mlContext.MulticlassClassification.Evaluate(predictions);

            Console.WriteLine($"Log-loss: {metrics.LogLoss}");
            Console.WriteLine($"Micro Accuracy: {metrics.MicroAccuracy}");
            Console.WriteLine($"Macro Accuracy: {metrics.MacroAccuracy}");
        }
    }
}
