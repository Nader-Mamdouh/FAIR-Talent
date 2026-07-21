namespace FAIR.Application.DTOs
{
    public record LoginResponse(bool Success = false
    , string message = null!
    , string Token = null!
    , string refreshToken = null!);
}
