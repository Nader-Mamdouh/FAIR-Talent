using AutoMapper;
using FAIR.Application.DTOs;
using FAIR.Application.DTOs.Identity;
using FAIR.Application.Services.Interfaces;
using FAIR.Application.Services.Interfaces.Logging;
using FAIR.Application.Validations;
using FAIR.Application.Validations.Authentication;
using FAIR.Domain.Entities.Identity;
using FAIR.Domain.Interfaces;
using FluentValidation;
using System.Security.Cryptography;
using System.Text;

namespace FAIR.Application.Services.Implementations.Authentication
{
    public class AuthenticationService(ITokenManagement tokenManagement
        , IUserRepository userRepository
        , IMapper mapper,
        IValidator<Register> registerValidation, IValidator<Login> loginUserValidation
        , IValidationService validation  ) : IAuthenticationService
    {
        private readonly IUserRepository _userRepository = userRepository;
        public async Task<ServiceResponse> CreateUser(Register createUser)
        {
            var validationResult = await validation.ValidateAsync(createUser, registerValidation);
            if (!validationResult.Success)
            {
                return validationResult;
            }
            var checkUserEmail = await _userRepository.GetByEmailAsync( createUser.Email );
            if ( checkUserEmail != null)
            {
                return  new ServiceResponse(false, "Email is already registered!");
            }
            var checkUserName = await _userRepository.GetByUsernameAsync(createUser.Username);
            if (checkUserName != null)
            {
                return new ServiceResponse(false, "UserName is already registered!");
            }
            AppUser user;
            if (createUser.Role.ToLower() == "player")
            {
                user = mapper.Map<Player>(createUser);
            }
            else if(createUser.Role.ToLower() == "coach")
            {
                user = mapper.Map<Coach>(createUser);
            }
            else
            {
                return new ServiceResponse(false, "This Role not available.");
            }
            //user.PasswordHash = HashPassword( createUser.Password);
            //await _userRepository.AddAsync(user);
            user.PasswordHash = createUser.Password;
           var isCreated =  await _userRepository.CreateUserAsync(user);
            return isCreated ? new ServiceResponse(true, "Created Account") :
                new ServiceResponse(false, "Error occure Create the Account");
            
        }

        public async Task<LoginResponse> LoginUser(Login login)
        {
            var _validationResult = await validation.ValidateAsync(login, loginUserValidation);
            if (!_validationResult.Success)
                return new LoginResponse(message: _validationResult.message);
            var user = await _userRepository.GetByUsernameAsync (login.Username);
            if (user == null)
                return new LoginResponse(message: "Invalid UserName or Password");

            //if (!VerifyPassword(login.Password, user.PasswordHash!))
            //{
            //    return new LoginResponse(message: "Invalid UserName or Password");
            //}
            
            var isValidPassword = await _userRepository.ChechPasswordAsync(user , login.Password);
            if (!isValidPassword)
            {
                return new LoginResponse(message: "Invalid UserName or Password");
            }

            string token = tokenManagement.GenerateToken(user);
            string refreshToken = tokenManagement.GetRefreshToken();
            int saveTokenResult = await tokenManagement.AddRefreshToken(user.Id, refreshToken);
            return saveTokenResult <= 0 ? new LoginResponse(message: "Internal error occurred while authentiacatint.") :
                new LoginResponse(Success: true, Token: token, refreshToken: refreshToken);
        }

        public async Task<LoginResponse> RevivToken(string refreshToken)
        {
            var validateTokenResult = await tokenManagement.ValidateRefreshToken(refreshToken);
            if (!validateTokenResult)
                return new LoginResponse(message: "Invalid Token");
            string UserId = await tokenManagement.GetUserIdByRefreshToken(refreshToken);
            var user = await _userRepository.GetByIdAsync(UserId);
            string newtoken = tokenManagement.GenerateToken(user);
            string newrefreshToken = tokenManagement.GetRefreshToken();
            await tokenManagement.UpdateRefreshToken(newrefreshToken);
            return new LoginResponse(Success: true, Token: newtoken, refreshToken: refreshToken);


        }
        private bool VerifyPassword(string password, string storedHash)
        {
            using var sha256 = SHA256.Create();
            var hashedBytes = sha256.ComputeHash(Encoding.UTF8.GetBytes(password));
            var hashedPassword = Convert.ToBase64String(hashedBytes);
            return hashedPassword == storedHash;
        }
        private string HashPassword(string password)
        {
            // In a real application, use a proper password hashing library like BCrypt
            // This is a simple example using SHA256
            using var sha256 = SHA256.Create();
            var hashedBytes = sha256.ComputeHash(Encoding.UTF8.GetBytes(password));
            return Convert.ToBase64String(hashedBytes);
        }
    }
}
