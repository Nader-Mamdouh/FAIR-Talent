using FAIR.Application.DTOs;
using FAIR.Application.DTOs.Profile;

namespace FAIR.Application.Services.Interfaces
{
    public interface IUserService
    {
        Task<PlayerProfile> GetPlayerProfileAsync(string playerId);
        Task<CoachProfile> GetCoachProfileAsync(string coachId);
        Task<ServiceResponse> UpdatePlayerProfileAsync(UpdatePlayerProfile profile);
        Task<ServiceResponse> UpdateCoachProfileAsync(UpdateCoachProfile profile);
        public Task<ServiceResponse> ChangePasswordAsync(string userId, ChangePasswordRequest request);
    }
}
