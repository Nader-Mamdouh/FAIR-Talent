using System.ComponentModel.DataAnnotations;
using System.Diagnostics.CodeAnalysis;

namespace FAIR.Application.DTOs.Profile
{
    public class UpdateCoachProfile
    {
        public required string Id { get; set; }

        [StringLength(100)]
        [NotNull]
        public required string FullName { get; set; }

        [StringLength(100)]
        [NotNull]
        public required string Specialization { get; set; }

        [Range(0, 70)]
        [NotNull]
        public int YearsOfExperience { get; set; }

        

    }
}
