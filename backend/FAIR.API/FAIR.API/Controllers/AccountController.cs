using FAIR.Application.DTOs.Profile;
using FAIR.Application.Services.Implementations.Profile;
using FAIR.Application.Services.Interfaces;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;

namespace FAIR.API.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    [Authorize]
    public class AccountController(IUserService _userService ) : ControllerBase
    {
        //private readonly UserService _userService = userService;
        [HttpGet("playerProfile/{playerId}")]
        public async Task<IActionResult> GetPlayerProfile(string playerId )
        {
            var profile = await _userService.GetPlayerProfileAsync(playerId);
            return profile != null? Ok(profile) : NotFound();
        }
        [HttpGet("coachProfile")]
        public async Task<IActionResult> CoachProfile(string coachId)
        {
            var profile = await _userService.GetCoachProfileAsync(coachId);
            return profile != null ? Ok(profile) : NotFound();
        }

        [HttpPut("updatePlayerProfile")]
        public async Task<IActionResult> UpdatePlayerProfile([FromBody] UpdatePlayerProfile playerProfile)
        {
            var result = await _userService.UpdatePlayerProfileAsync(playerProfile);
            return Ok(result);
        }
        [HttpPut("updateCoachProfile")]
        public async Task<IActionResult> UpdateCoachProfile([FromBody] UpdateCoachProfile coachProfile)
        {
            var result = await _userService.UpdateCoachProfileAsync(coachProfile);
            return Ok(result);
        }
        [HttpPut("change-password")]
        public async Task<IActionResult> ChangePassword( string userId, [FromBody]ChangePasswordRequest request )
        {
            var result = await _userService.ChangePasswordAsync(userId, request);
            return Ok(result);
        }
    }
}
