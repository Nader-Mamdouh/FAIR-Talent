using FAIR.Application.DTOs;
using FAIR.Application.DTOs.Identity;

namespace FAIR.Application.Services.Interfaces
{
    public interface IAuthenticationService
    {
        Task<ServiceResponse> CreateUser(Register createUser);
        Task<LoginResponse> LoginUser(Login login);
        Task<LoginResponse> RevivToken(string refreshToken);
    }
}
