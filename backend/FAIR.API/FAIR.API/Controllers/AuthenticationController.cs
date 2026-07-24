using FAIR.Application.DTOs.Identity;
using FAIR.Application.Services.Interfaces;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;

namespace FAIR.API.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class AuthenticationController(IAuthenticationService authentication ) : ControllerBase
    {
        [HttpPost("register")]
        public async Task<IActionResult> CreateUser(Register user)
        {
            var result = await authentication.CreateUser(user);
            if (result.Success)
                return Ok(result);
            return BadRequest(result);
        }
        [HttpPost("login")]
        public async Task<IActionResult> LoginUser(Login user)
        {
            var result = await authentication.LoginUser(user);
            if (result.Success)
                return Ok(result);
            return BadRequest(result);
        }
        [HttpGet("refreshToken/{refreshToken}")]
        public async Task<IActionResult> ReviveToken(string refreshToken)
        {
            var result = await authentication.RevivToken(refreshToken);
            if (result.Success)
                return Ok(result);
            return BadRequest(result);
        }
    }
}
