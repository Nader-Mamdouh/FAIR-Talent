using AutoMapper.QueryableExtensions;
using FAIR.Domain.Entities.Identity;
using FAIR.Domain.Interfaces;
using FAIR.Infrastructure.Data;
using Microsoft.AspNetCore.Identity;
using Microsoft.EntityFrameworkCore;
namespace FAIR.Infrastructure.Repository
{
    public class UserRepository : GenericRepository<AppUser> , IUserRepository
    {
        protected readonly UserManager<AppUser> _userManager;
        public UserRepository(AppDbContext context , UserManager<AppUser> userManager) : base(context)
        {
            _userManager = userManager;
        }
        public async Task<bool> CreateUserAsync(AppUser user )
        {
            //var _user = await _userManager.FindByEmailAsync(user.Email! );
            //if (_user != null) { return false; }

            return (await _userManager.CreateAsync(user!, user!.PasswordHash!)).Succeeded;
        }
        public async Task<bool> ChechPasswordAsync(AppUser user,string password)
        {   
            var result = await _userManager.CheckPasswordAsync(user, password!);
            return result;
        }

        public async Task<AppUser> GetByUsernameAsync(string username)
        {
            var userName = await context.Users
                .FirstOrDefaultAsync(u => u.UserName == username);
            return userName!;
        }

        public async Task<AppUser> GetByEmailAsync(string email)
        {
            var user= await _userManager.FindByEmailAsync(email!);
            return user!;
        }

        public async Task<Player> GetPlayerByIdAsync(string id)
        {
            var player = await context.Players
                .FirstOrDefaultAsync(p => p.Id == id);
            return player!;
        }

        public async Task<Coach> GetCoachByIdAsync(string id)
        {
            var coach= await context.Coaches
                .FirstOrDefaultAsync(c => c.Id == id);
            return coach!;
        }
    
        public async Task<IdentityResult> ChangePasswordAsync(string userId, string CurrentPassword , string NewPassword)
        {
            var user = await _userManager.FindByIdAsync(userId);

            return  await _userManager.ChangePasswordAsync(user!, CurrentPassword,NewPassword);

            //if (result.Succeeded)
            //    return true;

            //var error = result.Errors.First();

            //return Result.Failure(new Error(error.Code, error.Description, StatusCodes.Status400BadRequest));
        }

    }
}
