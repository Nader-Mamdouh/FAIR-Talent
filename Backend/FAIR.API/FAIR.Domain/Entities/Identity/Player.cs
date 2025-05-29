namespace FAIR.Domain.Entities.Identity
{
    public class Player : AppUser
    {
        public DateOnly DateOfBirth { get; set; }
        public string? Address { get; set; }
        public decimal Weight { get; set; }
        public decimal Height { get; set; }
        public  ICollection<Report>? Reports { get; set; }
    }
}
