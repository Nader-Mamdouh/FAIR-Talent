using FAIR.Domain.Entities;
using FAIR.Domain.Entities.Identity;
using FAIR.Domain.Interfaces;
using FAIR.Infrastructure.Data;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Internal;

namespace FAIR.Infrastructure.Repository
{
    public class ReportRepository: GenericRepository<Report>, IReportRepository 
    {
        public ReportRepository(AppDbContext _context) :base(_context)
        {
        }
        public async Task<IEnumerable<Report>> GetReportsByPlayerIdAsync(string playerId)
        {
            return await context.Reports
        //        .Include(r => r.Coach)
                .Where(r => r.PlayerId == playerId)
                .ToListAsync();
        }

        public async Task<IEnumerable<Report>> GetReportsByCoachIdAsync(string coachId)
        {
            return await context.Reports
                .Include(r => r.Player)
                .Where(r => r.CoachId == coachId)
                .ToListAsync();
        }

        public async Task<IEnumerable<Player>> GetTopPlayersByScoreAsync(int count)
        {
            // Get players with their reports and calculate average score
            var topPlayer =  await context.Players
                 .Include(p => p.Reports)
                 .OrderByDescending(p => p.Reports!.Average(r => r.ScorePercentage))
                 .ToListAsync();
            return topPlayer.Count() >= count ? topPlayer.Take(count) : topPlayer; 
          
            
                
        }
         
        public async Task<decimal> AverageScorePercentage(string playerId)
        {

            // calculate average score orecentage
            var result = await context.Reports.Where(p=>p.PlayerId == playerId).ToListAsync();
            return result.Any() ? result.Average(r => r.ScorePercentage) : 0;
                
                
        }
    }
}
