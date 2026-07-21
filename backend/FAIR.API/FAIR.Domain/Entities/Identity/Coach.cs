using System.Diagnostics.CodeAnalysis;

namespace FAIR.Domain.Entities.Identity
{
    public class Coach : AppUser
    {
        public  string? Specialization { get; set; }
        public int YearsOfExperience { get; set; }
        public  ICollection<Report>? Reports { get; set; }
    }
}
