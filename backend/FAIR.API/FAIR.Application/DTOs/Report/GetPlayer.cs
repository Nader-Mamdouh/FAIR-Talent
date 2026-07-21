using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using System.Runtime.CompilerServices;

namespace FAIR.Application.DTOs.Report
{
    public class GetPlayer
    {
        public string? Id { get; set; }
        public string? Username { get; set; }
        public string? Email { get; set; }
        public string? FullName { get; set; }
        public DateOnly DateOfBirth { get; set; }
        public string? Address { get; set; }
        public decimal AverageScore { get; set; }
    }
}
