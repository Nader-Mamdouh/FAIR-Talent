namespace FAIR.Application.DTOs.Profile
{
    public class PlayerProfile
    {
        public string Id { get; set; }
        public string Username { get; set; }
        public string Email { get; set; }
        public string FullName { get; set; }
        public DateOnly DateOfBirth { get; set; }
        public decimal Weight { get; set; }
        public decimal Height { get; set; }
        public string Address { get; set; }
    }
}
