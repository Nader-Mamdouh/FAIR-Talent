namespace FAIR.Application.DTOs
{
    public record ServiceResponse(bool Success = false, string message = null!);
}
