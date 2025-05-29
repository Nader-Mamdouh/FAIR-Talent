using System.ComponentModel.DataAnnotations;

namespace FAIR.Application.DTOs.Report
{
    public class ReportBase
    {
        public string? PlayerId { get; set; }
        public int Score { get; set; } // Player's performance score
        public decimal ScorePercentage { get; set; }
        public decimal AvgShotSpeed { get; set; }
        public decimal AvgSpeed { get; set; }
        public decimal MaxAcceleration { get; set; }
        public decimal MaxShotInconsistance { get; set; }
        public decimal MaxDistanceCovered { get; set; }
        public decimal MaxRallyContribution { get; set; }
        public string? Details { get; set; }
        public string? Comments { get; set; }
    }
}
