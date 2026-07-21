using AutoMapper;
using FAIR.Application.DTOs;
using FAIR.Application.DTOs.Profile;
using FAIR.Application.Services.Interfaces;
using FAIR.Domain.Entities.Identity;
using FAIR.Domain.Interfaces;
namespace FAIR.Application.Services.Implementations.Profile
{
    public class UserService(IUserRepository userRepository , IMapper mapper) : IUserService
    {
        public async Task<CoachProfile> GetCoachProfileAsync(string coachId)
        {
            var Coach = await userRepository.GetCoachByIdAsync( coachId );
            if( Coach == null )
                return new CoachProfile();
            return new CoachProfile
            {
                Id = Coach.Id,
                Username = Coach.UserName!,
                Email = Coach.Email!,
                FullName = Coach.FullName!,
                Specialization = Coach.Specialization!,
                YearsOfExperience = Coach.YearsOfExperience
            };

        }

        public async Task<PlayerProfile> GetPlayerProfileAsync(string playerId)
        {
            var Player = await userRepository.GetPlayerByIdAsync(playerId);
            if (Player == null)
                return new PlayerProfile();
            return new PlayerProfile
            {
                Id = Player.Id,
                Username = Player.UserName!,
                Email = Player.Email!,
                FullName = Player.FullName!,
                Address = Player.Address!,
                DateOfBirth = Player.DateOfBirth,
                Height = Player.Height,
                Weight =Player.Weight,

            };
        }

        public async Task<ServiceResponse> UpdateCoachProfileAsync(UpdateCoachProfile profile)
        {
            var coach = await userRepository.GetCoachByIdAsync(profile.Id);
            if (coach == null)
                return new ServiceResponse(false, "Coach Not Found!");
            //var updatePlayer = new Coach
            //{
            //    FullName = profile.FullName,
            //    YearsOfExperience = profile.YearsOfExperience,
            //    Specialization = profile.Specialization,
            //};
            coach.FullName = profile.FullName;
            coach.Specialization = profile.Specialization;
            coach.YearsOfExperience = profile.YearsOfExperience;

            var result = await userRepository.UpdateAsync(coach);
            return result > 0 ? new ServiceResponse(true, "Update Plofile!") :
                new ServiceResponse(false, "Error While Update Profile");
        }

        public async Task<ServiceResponse> UpdatePlayerProfileAsync(UpdatePlayerProfile profile)
        {
            var player = await userRepository.GetPlayerByIdAsync(profile.Id);
            if (player == null)
                return new ServiceResponse(false, "Player Not Found!");
            player.DateOfBirth = profile.DateOfBirth;
            player.Height = profile.Height;
            player.Weight = profile.Weight;
            player.FullName = profile.FullName;
            player.Address = profile.Address;

            var result= await userRepository.UpdateAsync(player);
            return result > 0 ? new ServiceResponse(true, "Update Plofile!"):
                new ServiceResponse(false, "Error While Update Profile");
        }
        public async Task<ServiceResponse> ChangePasswordAsync(string userId , ChangePasswordRequest request )
        {
            var result = await 
                userRepository.ChangePasswordAsync(userId, request.CurrentPassword , request.NewPassword);
            if (result.Succeeded)
                return new ServiceResponse(true, "Success to Change Password");

            return new ServiceResponse(false,result.Errors.First().Description);
        }
    }
}
