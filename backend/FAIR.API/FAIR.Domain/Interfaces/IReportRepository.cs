using FAIR.Domain.Entities.Identity;
using FAIR.Domain.Entities;
namespace FAIR.Domain.Interfaces
{
    public interface IReportRepository : IGenericRepository<Report>
    {
        Task<IEnumerable<Report>> GetReportsByPlayerIdAsync(string playerId);
        Task<IEnumerable<Report>> GetReportsByCoachIdAsync(string coachId);
        Task<IEnumerable<Player>> GetTopPlayersByScoreAsync(int count);
        public Task<decimal> AverageScorePercentage(string playerId);
    }
}
