using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLModel1_ConsoleApp1
{
    public class ModelTrainer
    {
        private readonly MLContext _mlContext;

        public ModelTrainer()
        {
            _mlContext = new MLContext();
        }

        public ITransformer TrainModel(IDataView trainingData)
        {
            var pipeline = _mlContext.Transforms.Concatenate("Features", "Edad", "Horas_Estudio_Semana", "Ranking_Acad_mico", "Horas_Sue_o", "Nivel_Estr_s")
                .Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "Exito_Acad_mico", maximumNumberOfIterations: 100))
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            return pipeline.Fit(trainingData);
        }

        public IDataView LoadData(string dataPath)
        {
            return _mlContext.Data.LoadFromTextFile<ModelInput>(
                dataPath,
                separatorChar: ',',
                hasHeader: true);
        }

        public (IDataView trainingData, IDataView testData) SplitData(IDataView dataView)
        {
            var split = _mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
            return (split.TrainSet, split.TestSet);
        }



        public ModelOutput Predict(ModelInput sampleData, ITransformer model)
        {
            var predEngine = _mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);
            return predEngine.Predict(sampleData);
        }
    }
    }
