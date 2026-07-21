using FAIR.Domain.Entities.Identity;
using System.Security.Claims;

namespace FAIR.Domain.Interfaces
{
    public interface ITokenManagement
    {
        string GenerateToken(AppUser user);
        Task<int> AddRefreshToken(string userId, string RefreshToken);
        Task<int> UpdateRefreshToken(string refreshToken);
        string GetRefreshToken();
        public  Task<string> GetUserIdByRefreshToken(string refreshToken);
        Task<bool> ValidateRefreshToken(string refreshToken);
    }
}
