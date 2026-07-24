namespace FAIR.Application.DTOs.Identity
{
    public class Register : BaseModel
    {
        public required string Email { get; set; }
        public required string FullName { get; set; }
        public required string Role { get; set; }
        public required DateOnly DateOfBirth { get; set; }
        public required string Location { get; set; }
        public string? Specialization { get; set; }
        public int? YearsOfExperience { get; set; }
    }

}
