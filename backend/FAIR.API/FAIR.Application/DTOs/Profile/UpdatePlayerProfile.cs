using System.ComponentModel.DataAnnotations;

namespace FAIR.Application.DTOs.Profile
{
    public class UpdatePlayerProfile
    {
        public required string Id { get; set; }

        [StringLength(100)]
        public required string FullName { get; set; }

        public required DateOnly DateOfBirth { get; set; }

        [Range(0, 250)]
        public required decimal Weight { get; set; }

        [Range(0, 250)]
        public required decimal Height { get; set; }

        [StringLength(200)]
        public required string Address { get; set; }

    }
}
