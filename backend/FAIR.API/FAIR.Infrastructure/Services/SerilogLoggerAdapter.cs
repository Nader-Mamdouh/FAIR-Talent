using FAIR.Application.Services.Interfaces.Logging;
using Microsoft.Extensions.Logging;
namespace eCommerce.Infrastructure.Services
{
    public class SerilogLoggerAdapter<T>(ILogger<T> logger) : IAppLogger<T>
    {
        public void LogError(Exception ex, string message) => logger.LogError(ex, message);

        public void LogInformation(string message) => logger.LogInformation(message);

        public void LogWarning(string message) => logger.LogWarning(message);
    }
}
