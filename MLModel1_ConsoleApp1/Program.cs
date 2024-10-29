

namespace MLModel1_ConsoleApp1
{
    public class Program
    {
        private static void Main(string[] args)
        {
            // Inicializar clases
            var modelTrainer = new ModelTrainer();
            var modelEvaluator = new ModelEvaluator();

            // Cargar datos desde un archivo CSV
            string dataPath = @"C:\Users\dotel\Downloads\Exito-academico-practica4.csv"; 
            var dataView = modelTrainer.LoadData(dataPath);

            // Dividir los datos en conjunto de entrenamiento y prueba
            var (trainingData, testData) = modelTrainer.SplitData(dataView);

            // Entrenar el modelo
            var model = modelTrainer.TrainModel(trainingData);

            // Evaluar el modelo
            modelEvaluator.EvaluateModel(model, testData);

            // Realizar una predicción con datos de muestra
            var sampleData = new ModelInput()
            {
                ID_Estudiante = 2F,
                Edad = 19F,
                Horas_Estudio_Semana = 5F,
                Ranking_Acad_mico = 82F,
                Horas_Sue_o = 5F,
                Nivel_Estr_s = 9F,
            };

            var prediction = modelTrainer.Predict(sampleData, model);
            Console.WriteLine($"Predicción: {prediction.Exito_Acad_mico}");

            // Mostrar scores de cada clase
            Console.WriteLine($"Scores for each class:");
            for (int i = 0; i < prediction.Score.Length; i++)
            {
                Console.WriteLine($"Class {i}: {prediction.Score[i]}");
            }

            Console.WriteLine("=============== End of process, hit any key to finish ===============");
            Console.ReadKey();
        }
    }
}
