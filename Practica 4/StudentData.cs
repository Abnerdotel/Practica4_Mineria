using Microsoft.ML.Data;


namespace Practica_4
{
    public class StudentData
    {
        [LoadColumn(0)] 
        public int ID_Estudiante { get; set; }

        [LoadColumn(1)] 
        public float Edad { get; set; }

        [LoadColumn(2)] 
        public float Horas_Estudio_Semana { get; set; }

        [LoadColumn(3)] 
        public float Ranking_Academico { get; set; }

        [LoadColumn(4)] 
        public float Horas_Sueno { get; set; }

        [LoadColumn(5)] 
        public float Nivel_Estres { get; set; }

        [LoadColumn(6)] 
        public float Exito_Academico { get; set; }
    }

}
