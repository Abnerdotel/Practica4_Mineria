using System;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Practica_4
{
    public class ModelTrainer
    {
        private readonly MLContext _mlContext;
        private ITransformer _model;

        public ModelTrainer()
        {
            _mlContext = new MLContext();
        }

        // Método para cargar y dividir los datos
        public (IDataView TrainingData, IDataView TestData) LoadData(string dataPath)
        {
            IDataView dataView = _mlContext.Data.LoadFromTextFile<StudentData>(
                dataPath, hasHeader: true, separatorChar: '\t'); // Cambia ',' a '\t' si usas tabulaciones

            var dataSplit = _mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
            return (dataSplit.TrainSet, dataSplit.TestSet);
        }

        // Método para definir el pipeline y entrenar el modelo
        public void TrainModel(IDataView trainingData)
        {
            var pipeline = _mlContext.Transforms.Conversion.MapValueToKey("Label", nameof(StudentData.Exito_Academico))
                .Append(_mlContext.Transforms.Concatenate("Features", nameof(StudentData.Edad), nameof(StudentData.Horas_Estudio_Semana), nameof(StudentData.Ranking_Academico), nameof(StudentData.Horas_Sueno), nameof(StudentData.Nivel_Estres)))
                .Append(_mlContext.Transforms.NormalizeMinMax("Features"))
                .Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy())
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            _model = pipeline.Fit(trainingData);
        }

        // Método para evaluar el modelo
        public void EvaluateModel(IDataView testData)
        {
            var predictions = _model.Transform(testData);
            var metrics = _mlContext.MulticlassClassification.Evaluate(predictions);

            Console.WriteLine($"MicroAccuracy: {metrics.MicroAccuracy:F2}");
            Console.WriteLine($"MacroAccuracy: {metrics.MacroAccuracy:F2}");
            Console.WriteLine($"LogLoss: {metrics.LogLoss:F2}");
        }

        // Método para hacer predicciones
        public StudentPrediction Predict(StudentData studentData)
        {
            var predEngine = _mlContext.Model.CreatePredictionEngine<StudentData, StudentPrediction>(_model);
            return predEngine.Predict(studentData);
        }
    }
}

