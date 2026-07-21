using Microsoft.AspNetCore.Identity;
namespace FAIR.Domain.Entities.Identity
{
    public abstract class AppUser : IdentityUser
    {
        //public string Id { get; set; } = Guid.NewGuid().ToString();
       // public string? Username { get; set; }
       // public string? Email { get; set; }
        public string? FullName { get; set; }
        //public string? PasswordHash { get; set; }
    }
}
