namespace Practica_4
{
    class Program
    {
        static void Main(string[] args)
        {
            //  entrenador de modelo
            var modelTrainer = new ModelTrainer();

            // Cargar y dividir los datos
            string dataPath = @"C:\Users\dotel\Downloads\Exito-academico-practica4.csv";
            var (trainingData, testData) = modelTrainer.LoadData(dataPath);

            // Entrenar el modelo
            modelTrainer.TrainModel(trainingData);

            // Evaluar el modelo
            modelTrainer.EvaluateModel(testData);

            // Hacer una predicción de ejemplo
            var newStudent = new StudentData
            {
                Edad = 21F,
                Horas_Estudio_Semana = 15F,
                Ranking_Academico = 120F, 
                Horas_Sueno = 7F, 
                Nivel_Estres = 5F 
            };

            var prediction = modelTrainer.Predict(newStudent);
            Console.WriteLine($"Predicción: {prediction.PredictedLabel}"); 
        }
    }
}
