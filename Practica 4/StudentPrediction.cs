
using Microsoft.ML.Data;

namespace Practica_4
{
    public class StudentPrediction
    {
        [ColumnName("PredictedLabel")] 
        public string PredictedLabel { get; set; } 

        public float Probability { get; set; } 
        public float Score { get; set; } 
    }
}
