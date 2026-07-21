using System.ComponentModel.DataAnnotations;

namespace FAIR.Application.DTOs.Report
{
    public class GetReport : ReportBase
    {
        [Required]
        [Key]
        public string? Id { get; set; }
    }
}
