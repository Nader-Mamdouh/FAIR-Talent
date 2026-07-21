using FAIR.Application.DTOs;
using FAIR.Application.DTOs.Report;
namespace FAIR.API.Interfaces.Report
{
    public interface IReportService
    {
        Task<ServiceResponse> AddAsync(CreateReport report);
        Task<ServiceResponse> UpdateAsync(UpdateReport report);
        Task<ServiceResponse> DeleteAsync(string Id);
        Task<IEnumerable<GetReport>> GetAllAsync();
        Task<GetReport> GetByIdAsync(string id);
        Task<IEnumerable<GetReport>> GetReportsByPlayerIdAsync(string playerId);
        Task<IEnumerable<GetPlayer>> GetTopPlayersByScoreAsync(int count);
    }
}
