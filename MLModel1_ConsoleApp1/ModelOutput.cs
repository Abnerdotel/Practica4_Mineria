using Microsoft.ML.Data;


namespace MLModel1_ConsoleApp1
{
    public class ModelOutput
    {
        [ColumnName("PredictedLabel")]
        public string Exito_Acad_mico { get; set; }

        public float[] Score { get; set; }
    }
}
