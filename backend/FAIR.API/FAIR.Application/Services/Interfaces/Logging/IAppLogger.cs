namespace FAIR.Application.Services.Interfaces.Logging
{
    public interface IAppLogger<T>
    {
        public void LogError(Exception ex, string message);
        public void LogWarning(string message);
        public void LogInformation(string message);
    }
}
