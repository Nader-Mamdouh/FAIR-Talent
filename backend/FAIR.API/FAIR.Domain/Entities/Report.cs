using FAIR.Domain.Entities.Identity;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
namespace FAIR.Domain.Entities
{
    public class Report
    {
        [Key]
        public string Id { get; set; } = Guid.NewGuid().ToString();
        public string? PlayerId { get; set; }
        public string? CoachId { get; set; }
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
        public DateTime CreatedDate { get; set; }
        public DateTime? UpdatedDate { get; set; }
        [ForeignKey(nameof(PlayerId))]
        public  Player? Player { get; set; }
        [ForeignKey(nameof(CoachId))]
        public Coach? Coach { get; set; }
       
    }
}
