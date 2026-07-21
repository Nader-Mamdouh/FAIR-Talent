using FAIR.Domain.Entities.Identity;
using Microsoft.AspNetCore.Identity;
using System.Security.Claims;

namespace FAIR.Domain.Interfaces
{
    public interface IUserRepository : IGenericRepository<AppUser>
    {
        Task<bool> CreateUserAsync(AppUser user );
        Task<bool> ChechPasswordAsync(AppUser user , string password);
        Task<AppUser> GetByUsernameAsync(string username);
        Task<AppUser> GetByEmailAsync(string email);
        Task<Player> GetPlayerByIdAsync(string id);
        Task<Coach> GetCoachByIdAsync(string id);
        Task<IdentityResult> ChangePasswordAsync(string userId, string CurrentPassword, string NewPassword);


    }
}
